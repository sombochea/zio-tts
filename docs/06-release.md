# 06 — Release to Hugging Face / GitHub

Publishing runs **on Modal**, using the `huggingface-secret` Modal secret (no local HF login needed). It stages an
inference-only checkpoint (optimizer state dropped: 998 MB -> 333 MB), ONNX export, `vocab.json`, four generated
samples and the model card, then uploads.

```bash
modal deploy scripts/modal_train.py                       # after any code/doc change
RUN=/vol/outputs/tts_v0/<run_dir>                        # path on the zio-data volume
uv run python scripts/modal_publish.py $RUN <user>/zio-tts              # DRY RUN: lists files, checks auth
uv run python scripts/modal_publish.py $RUN <user>/zio-tts --push       # uploads as PRIVATE
uv run python scripts/modal_publish.py $RUN <user>/zio-tts --push --public   # public
```
(`modal app logs zio-train` shows the remote output if your terminal does not.)

## Before making a repo public
1. Listen to `samples/` and to your own test sentences.
2. Confirm the license in `docs/model_cards/tts.md` (CC BY-SA 4.0 because SLR42 is share-alike; not legal advice).
3. Add numbers (intelligibility CER via our STT, small MOS panel) to the card's *Evaluation* section.
4. Only release voices you have consent for; state that outputs are synthetic.

## STT
`docs/model_cards/stt.md` is a **draft**: no STT model has been trained at scale or evaluated. Train with
`configs/stt_small.yaml` on Modal ([04](04-stt-training.md)), add CER/WER, then extend `publish` for STT.
