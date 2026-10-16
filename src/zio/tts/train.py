"""VITS (coqui-tts) on graphemes, Khmer + English.

uv run --extra tts python -m zio.tts.train configs/tts_tiny.yaml
"""
import json
import sys
from functools import partial
from pathlib import Path

import yaml
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from zio.tts.vocab import CHARS, PUNCT


def voice_id(r: dict, emotions: bool) -> str:
    # emotion is just part of the voice id: one embedding per (speaker, emotion)
    return f'{r["speaker"]}__{r.get("emotion", "neutral")}' if emotions else r["speaker"]


def manifest_formatter(root_path, meta_file, ignored_speakers=None, emotions=False, **kw):
    out = []
    for line in Path(root_path, meta_file).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.append({"text": r["text"].lower(), "audio_file": str(Path(root_path, r["audio"])),
                    "speaker_name": voice_id(r, emotions), "raw_speaker": r["speaker"], "root_path": str(root_path), "language": r["lang"]})
    return out


def main(path: str):
    c = yaml.safe_load(open(path))
    ds = BaseDatasetConfig(formatter="", dataset_name="zio", path=c["data"], meta_file_train="manifest.jsonl")
    cfg = VitsConfig(
        audio=VitsAudioConfig(sample_rate=24000, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None),
        run_name=c["run"], output_path=c["out"], batch_size=c["batch"], eval_batch_size=c["batch"],
        batch_group_size=0, num_loader_workers=c.get("workers", 2), num_eval_loader_workers=1,
        run_eval=True, test_delay_epochs=-1, epochs=c["epochs"], text_cleaner="basic_cleaners", use_phonemes=False,
        compute_input_seq_cache=True, print_step=c.get("print_step", 25), print_eval=True,
        mixed_precision=c.get("fp16", False), save_step=c.get("save_step", 1000), save_n_checkpoints=2,
        max_text_len=c.get("max_text_len", 300), min_audio_len=24000, max_audio_len=24000 * 15,
        characters=CharactersConfig(characters_class="TTS.tts.utils.text.characters.Graphemes", pad="<PAD>",
                                    eos="<EOS>", bos="<BOS>", blank="<BLNK>", characters=CHARS, punctuations=PUNCT + " ",
                                    phonemes=None),
        test_sentences=c.get("test_sentences", []), datasets=[ds], cudnn_benchmark=False)
    ap = AudioProcessor.init_from_config(cfg)
    tok, cfg = TTSTokenizer.init_from_config(cfg)
    train, evl = load_tts_samples(ds, eval_split=True, formatter=partial(manifest_formatter, emotions=c.get("voices", False)),
                                  eval_split_max_size=c.get("eval_size", 50), eval_split_size=0.02)
    allx = train + evl
    if c.get("speakers"):  # v0 is single-voice: mixing speakers blurs the voice. Filter BEFORE splitting.
        allx = [x for x in allx if x["raw_speaker"] in set(c["speakers"])]
    if c.get("limit"):
        allx = allx[:c["limit"]]
    n_eval = min(c.get("eval_size", 20), max(len(allx) // 10, 1))
    train, evl = allx[:-n_eval], allx[-n_eval:]
    sm = None
    if c.get("voices"):  # multi-voice: one embedding per (speaker, emotion); chosen by id at inference
        sm = SpeakerManager()
        sm.set_ids_from_data(allx, parse_key="speaker_name")
        cfg.model_args.use_speaker_embedding = True
        cfg.model_args.num_speakers = sm.num_speakers
        cfg.num_speakers = sm.num_speakers
    model = Vits(cfg, ap, tok, speaker_manager=sm)
    Trainer(TrainerArgs(), cfg, c["out"], model=model, train_samples=train, eval_samples=evl).fit()


if __name__ == "__main__":
    main(sys.argv[1])
