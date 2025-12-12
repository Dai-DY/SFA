SFA - Step 1: VLM + SAM target localization

This repo provides a minimal runnable scaffold for Step 1: parse a language instruction to a target word and produce a 2D mask using SAM (placeholder).

Files:
- `src/parser.py`: instruction parser (uses OpenAI if `OPENAI_API_KEY` set, otherwise heuristic)
- `src/segmenter.py`: placeholder SAM wrapper. Replace with Grounded-SAM/MobileSAM integration. Set `USE_GROUNDED_SAM=1` to enable real model integration point.
- `src/cli.py`: simple CLI to run the pipeline and save a mask overlay.

How to run a quick smoke test (no heavy models):

1. Install deps: `pip install -r requirements.txt`
2. Run: `python -m src.cli path/to/image.jpg "Please pick up the carrot on the table."`

To integrate Grounded-SAM / MobileSAM:

1. Implement model loading and inference in `src/segmenter.py:segment_target`.
2. When calling Grounded-SAM, pass the target word as text prompt and return the binary mask and bbox.
