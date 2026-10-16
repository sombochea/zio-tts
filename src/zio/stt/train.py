"""Whisper LoRA fine-tune for Khmer + English.

uv run --extra stt python -m zio.stt.train configs/stt_tiny.yaml
"""
import sys
from pathlib import Path
from dataclasses import dataclass

import jiwer
import torch
import yaml
from datasets import concatenate_datasets
from peft import LoraConfig, get_peft_model
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor)

from zio.stt.data import NAMES, from_fleurs, from_manifest, load_audio


def build_split(cfg: dict, split: str):
    fsplit = "train" if split == "train" else "validation"
    parts = [from_fleurs(l, fsplit, cfg.get("limit")) for l in cfg.get("fleurs", [])]
    key = "manifests" if split == "train" else "eval_manifests"
    parts += [from_manifest(Path(m)) for m in cfg.get(key, [])]
    if cfg.get("limit"):
        parts = [p.select(range(min(cfg["limit"], len(p)))) for p in parts]
    return concatenate_datasets(parts).shuffle(seed=0)


@dataclass
class Collator:
    proc: WhisperProcessor

    def __call__(self, rows):
        tok = self.proc.tokenizer
        feats = self.proc.feature_extractor([load_audio(r["audio"]) for r in rows], sampling_rate=16000,
                                            return_tensors="pt")
        ids = []
        for r in rows:
            tok.set_prefix_tokens(language=NAMES[r["lang"]], task="transcribe")
            ids.append(tok(r["text"]).input_ids)
        n = max(map(len, ids))
        labels = torch.full((len(ids), n), -100)
        for i, x in enumerate(ids):
            labels[i, :len(x)] = torch.tensor(x)
        # decoder input is labels shifted by the model; drop the leading <sot> as HF recipe does
        if (labels[:, 0] == self.proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")).all():
            labels = labels[:, 1:]
        return {"input_features": feats.input_features, "labels": labels}


def main(path: str):
    cfg = yaml.safe_load(open(path))
    proc = WhisperProcessor.from_pretrained(cfg["model"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model"])
    model.generation_config.forced_decoder_ids = None
    model = get_peft_model(model, LoraConfig(r=cfg.get("lora_r", 32), lora_alpha=64, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]))
    model.print_trainable_parameters()
    train, evald = build_split(cfg, "train"), build_split(cfg, "validation")

    def metrics(p):
        tok = proc.tokenizer
        p.label_ids[p.label_ids == -100] = tok.pad_token_id
        pred = tok.batch_decode(p.predictions, skip_special_tokens=True)
        ref = tok.batch_decode(p.label_ids, skip_special_tokens=True)
        return {"cer": jiwer.cer(ref, pred), "wer": jiwer.wer(ref, pred)}

    args = Seq2SeqTrainingArguments(
        output_dir=cfg["out"], per_device_train_batch_size=cfg["batch"], per_device_eval_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg.get("accum", 1), learning_rate=float(cfg["lr"]), max_steps=cfg["steps"],
        warmup_steps=cfg.get("warmup", 50), eval_strategy="steps", eval_steps=cfg["eval_every"],
        save_steps=cfg["eval_every"], logging_steps=10, predict_with_generate=True, generation_max_length=225,
        bf16=torch.cuda.is_available(), use_cpu=cfg.get("cpu", False), remove_unused_columns=False, label_names=["labels"],
        report_to=cfg.get("report_to", "none"), save_total_limit=2)
    Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=evald, data_collator=Collator(proc),
                   compute_metrics=metrics).train()
    model = model.merge_and_unload()
    model.save_pretrained(f"{cfg['out']}/merged")
    proc.save_pretrained(f"{cfg['out']}/merged")


if __name__ == "__main__":
    main(sys.argv[1])
