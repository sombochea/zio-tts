"""Run any zio trainer on a Modal GPU.

  modal volume put zio-data data/processed /processed      # once; creates the volume if needed
  uv run modal run scripts/modal_train.py --config configs/stt_small.yaml
  uv run modal run scripts/modal_train.py --config configs/tts_base.yaml --module zio.tts.train
  modal volume get zio-data /outputs/stt_small ./outputs    # fetch the merged model
"""
import modal

app = modal.App("zio-train")
vol = modal.Volume.from_name("zio-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("espeak-ng")
         .uv_pip_install("torch", "transformers", "datasets", "peft", "accelerate", "jiwer", "soundfile",
                         "soxr", "numpy", "pyyaml", "khmer-nltk", "coqui-tts[codec]", "torchaudio")
         .add_local_python_source("zio")
         .add_local_dir("configs", "/root/configs"))


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
    cfg["out"] = "/vol/" + cfg["out"]
    yaml.safe_dump(cfg, open("/tmp/cfg.yaml", "w"))
    sys.argv = [module, "/tmp/cfg.yaml"]
    runpy.run_module(module, run_name="__main__")
    vol.commit()


@app.local_entrypoint()
def main(config: str, module: str = "zio.stt.train"):
    train.remote(config, module)
