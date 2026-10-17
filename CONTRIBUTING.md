# Contributing

The most useful contribution is **Khmer speech data**, especially expressive speech: emotions, laughter, conversation. Code fixes are welcome too.

## Share a dataset
1. **Consent.** Every speaker agrees, in writing, to their voice being used for training and released openly. Pick a license that allows that (CC BY 4.0 or CC BY-SA 4.0).
2. **Record** following [docs/02-recording.md](docs/02-recording.md). Put the wavs in a folder with a `metadata.jsonl`:
   ```json
   {"file_name": "001.wav", "text": "ខ្ញុំសប្បាយចិត្តណាស់!", "speaker": "dara", "emotion": "happy", "events": ["laugh"]}
   ```
3. **Check and prepare** into the dataset format (24 kHz, trimmed, normalized, word-spaced):
   ```bash
   uv run python -m zio.data.qc recordings/*.wav
   uv run python -m zio.data.prepare folder data/processed/<name>_km --src recordings --source "<you or URL>" --license cc-by-4.0
   ```
4. **Open a pull request** on the Hub dataset: `uv run python scripts/push_dataset.py data/processed/<name>_km --push --pr`. In a GitHub PR, add your folder to `docs/cards/dataset.md` (`configs:` and the table).

Already have a public corpus on the Hub? Add one line to `CORPORA` in [`src/zio/data/prepare.py`](src/zio/data/prepare.py) ([how](docs/01-data.md#add-a-source)).

## Code
```bash
uv sync --extra tts --extra stt && uv run pytest
```
Keep changes small and match the surrounding style. Put the reason in the docs page that owns the topic.
