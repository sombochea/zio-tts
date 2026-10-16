"""Run any zio trainer on a Modal GPU.

  modal volume create zio-data && modal volume put zio-data data/processed /processed   # once

Long runs: DEPLOY then SPAWN, so the job does not depend on any local process
(a plain `modal run` is cancelled if the local client dies, even with --detach):
  modal deploy scripts/modal_train.py            # redo after any code/config change
  uv run python scripts/modal_spawn.py configs/tts_base.yaml zio.tts.train
  modal app logs zio-train                       # watch;  modal app stop zio-train  # cancel

Short runs: uv run modal run scripts/modal_train.py --config configs/stt_tiny.yaml
  modal volume get zio-data /outputs/stt_small ./outputs    # fetch the merged model
"""
import modal

app = modal.App("zio-train")
vol = modal.Volume.from_name("zio-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("ffmpeg")  # torchcodec (audio IO) needs it
         .uv_pip_install("torch", "transformers", "datasets", "peft", "accelerate", "jiwer", "soundfile",
                         "soxr", "numpy", "pyyaml", "khmer-nltk", "coqui-tts[codec]", "torchaudio", "onnx", "onnxscript", "onnxruntime")
         .env({"PYTHONPATH": "/root"})
         .add_local_dir("src/zio", "/root/zio")
         .add_local_dir("configs", "/root/configs")
         .add_local_dir("docs/model_cards", "/model_cards")
         .add_local_dir("examples", "/root/examples"))


@app.function(image=image, gpu="A100-80GB", timeout=24 * 3600, secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache})
def train(config: str, module: str = "zio.stt.train"):
    import os, runpy, sys, yaml
    cfg = yaml.safe_load(open(config))
    # remap local paths onto the volume
    cfg["manifests"] = [m.replace("data/processed", "/vol/processed") for m in cfg.get("manifests", [])]
    cfg["eval_manifests"] = [m.replace("data/processed", "/vol/processed") for m in cfg.get("eval_manifests", [])]
    if "data" in cfg:
        cfg["data"] = cfg["data"].replace("data/processed", "/vol/processed")
    if cfg.get("resume"):  # path under outputs/ on the volume
        cfg["resume"] = "/vol/" + cfg["resume"]
    cfg["out"] = "/vol/" + cfg["out"]
    yaml.safe_dump(cfg, open("/tmp/cfg.yaml", "w"))
    import threading, time

    def autosave():  # volume writes are only visible/durable after commit
        while True:
            time.sleep(600)
            vol.commit()

    threading.Thread(target=autosave, daemon=True).start()
    sys.argv = [module, "/tmp/cfg.yaml"]
    runpy.run_module(module, run_name="__main__")
    vol.commit()


@app.local_entrypoint()
def main(config: str, module: str = "zio.stt.train"):
    train.remote(config, module)


SAMPLES = ["សួស្តី តើអ្នកសុខសប្បាយជាទេ", "ថ្ងៃនេះអាកាសធាតុល្អណាស់", "ម្ហូបនេះតម្លៃ ៥០០០៛", "កម្ពុជាជាប្រទេសមានប្រវត្តិសាស្ត្រយូរលង់"]


@app.function(image=image, cpu=4, memory=16384, timeout=3600, secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/vol": vol})
def publish(run_dir: str, repo: str, kind: str = "tts", push: bool = False, public: bool = False):
    """Stage a TTS run (inference-only ckpt, ONNX, vocab, samples, card) and optionally upload to the HF Hub.
    Dry run unless push=True; repos are private unless public=True."""
    import os, shutil
    from pathlib import Path

    import soundfile  # noqa: F401
    import torch
    from huggingface_hub import HfApi

    from zio.tts.export_onnx import export, write_vocab
    from zio.tts.infer import synthesize

    token = next((os.environ[k] for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN") if k in os.environ), None)
    assert token, f"no HF token in secret; env keys: {[k for k in os.environ if 'HF' in k or 'HUG' in k.upper()]}"
    api = HfApi(token=token)
    print("authenticated as", api.whoami()["name"])
    src, st = Path(run_dir), Path("/tmp/stage")
    shutil.rmtree(st, ignore_errors=True)
    (st / "samples").mkdir(parents=True)
    torch.save({"model": torch.load(src / "best_model.pth", map_location="cpu", weights_only=False)["model"]},
               st / "best_model.pth")  # drop optimizer state; coqui's loader still needs disc.* weights
    shutil.copy(src / "config.json", st / "config.json")
    export(str(src), str(st / "zio-tts.onnx"))
    write_vocab(str(src), str(st / "vocab.json"))
    for i, t in enumerate(SAMPLES, 1):
        synthesize(str(src), t, str(st / "samples" / f"sample{i}.wav"))
    card = Path("/model_cards/tts.md").read_text().replace("{{REPO}}", repo)
    card += "\n## Sample sentences\n" + "\n".join(f"{i}. {t} — `samples/sample{i}.wav`" for i, t in enumerate(SAMPLES, 1)) + "\n"
    (st / "README.md").write_text(card)
    for f in sorted(st.rglob("*")):
        if f.is_file():
            print(f"{f.stat().st_size / 1e6:9.1f} MB  {f.relative_to(st)}")
    if not push:
        print(f"DRY RUN: would upload to {repo} ({'public' if public else 'private'})")
        return
    api.create_repo(repo, private=not public, exist_ok=True)
    api.upload_folder(folder_path=str(st), repo_id=repo, commit_message="Zio TTS v0")
    print("uploaded", repo)


@app.function(image=image, cpu=4, memory=16384, timeout=1800, secrets=[modal.Secret.from_name("huggingface-secret")])
def verify(repo: str):
    """Download a published repo fresh and run the card's two getting-started paths (Python + ONNX)."""
    import os, sys

    import onnxruntime  # noqa: F401
    import soundfile as sf
    from huggingface_hub import HfApi, snapshot_download

    token = next(os.environ[k] for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN") if k in os.environ)
    print("repo files:", sorted(HfApi(token=token).list_repo_files(repo)))
    d = snapshot_download(repo, token=token, local_dir="/tmp/dl")
    from zio.tts.infer import synthesize
    synthesize(d, "សួស្តី តើអ្នកសុខសប្បាយជាទេ", "/tmp/py.wav")
    sys.path.insert(0, "/root")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ox", "/root/examples/onnx_infer.py")
    ox = importlib.util.module_from_spec(spec); spec.loader.exec_module(ox)
    a = ox.synth(f"{d}/zio-tts.onnx", f"{d}/vocab.json", "សួស្តី តើអ្នកសុខសប្បាយជាទេ")
    b, sr = sf.read("/tmp/py.wav")
    print(f"VERIFY python: {len(b) / sr:.2f}s  onnx: {len(a) / 24000:.2f}s  (should be close)")
