# 08 — Train on Kaggle (free GPU) and publish with `hf`

Kaggle gives a free GPU (T4/P100, ~9 h per session, ~30 h/week). The kernel resumes `tts_v1`, stops at the time budget, then
stages and uploads with the `hf` CLI. Uploads are **private** unless you add `--public` yourself.

## One-time setup
```bash
uv tool install kaggle huggingface_hub     # `kaggle` and `hf` CLIs
kaggle auth login                          # browser OAuth (or: export KAGGLE_API_TOKEN=...)
hf auth login                              # write token
```

## Launch
```bash
# uploads data + the Modal interim checkpoint as a private dataset, pushes the kernel (GPU + internet on)
uv run python scripts/kaggle_launch.py data/processed/km_male --ckpt outputs/tts_v1_interim/best_model.pth
```
Then, once, in the kernel page (Add-ons -> Secrets) add secret `HF_TOKEN` (a Hugging Face write token) and run it.
Without the secret the kernel still trains and leaves the staged release in its output.

```bash
kaggle kernels status <user>/zio-tts-train
kaggle kernels logs   <user>/zio-tts-train
kaggle kernels output <user>/zio-tts-train -p kaggle_out      # checkpoints + staged files
```
No secret? Publish from your machine: `uv run --extra tts python scripts/push_to_hf.py kaggle_out/out/<run> <user>/zio-tts --push`.

## Notes
- Kaggle's GPU is slower than an A100 and batch is 16 (not 32), so reaching 450 epochs takes several sessions. To continue: download
  the output, upload the newest `best_model*.pth` as the dataset checkpoint again (`--ckpt`), relaunch.
- `ZIO_HOURS` (default 8.5) is the training time budget; `ZIO_REPO` the target repo.
- The kernel installs packages with pip on Kaggle's image (not `uv.lock`). If a Kaggle image update breaks it, check `kaggle kernels logs`.
- Untested end to end: no Kaggle login existed on the machine that wrote this.
