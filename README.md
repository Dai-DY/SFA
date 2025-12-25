SFA - Step 1: VLM + SAM target localization

This repo provides a minimal runnable scaffold for Step 1: parse a language instruction to a target word and produce a 2D mask using SAM (placeholder).

Files:
- `src/parser.py`: instruction parser (uses OpenAI if `OPENAI_API_KEY` set, otherwise heuristic)
- `src/segmenter.py`: placeholder SAM wrapper. Replace with Grounded-SAM/MobileSAM integration. Set `USE_GROUNDED_SAM=1` to enable real model integration point.
	- Updated: `scripts/test_videos.py` now uses `src.get_target.extract_action_objects` to extract both `source_object` and `destination_object` from an instruction and overlays both masks (different colors) when available.
	- `src/segmenter.py` tries to import Grounded-SAM plugins when `USE_GROUNDED_SAM` is set and otherwise falls back to a placeholder circular mask.
- `src/cli.py`: simple CLI to run the pipeline and save a mask overlay.

How to run a quick smoke test (no heavy models):

1. Install deps: `pip install -r requirements.txt`
2. Run: `python -m src.cli path/to/image.jpg "Please pick up the carrot on the table."`

To integrate Grounded-SAM / MobileSAM:

1. Implement model loading and inference in `src/segmenter.py:segment_target`.
2. When calling Grounded-SAM, pass the target word as text prompt and return the binary mask and bbox.
3. To run the sample video overlay script (uses `get_target` and overlays both source/destination masks):

```bash
# run within SFA environment; if using real grounded-sam set USE_GROUNDED_SAM=1
python scripts/test_videos.py
```

Notes:
- Set `USE_GROUNDED_SAM=1` to attempt to use the Grounded-SAM plugin. If no plugin is found, the script falls
	back to the placeholder circular mask.
- If you plan to use a specific Grounded-SAM integration (e.g., a plugin or wrapper), ensure its module
	name is one of the common ones or adapt `src/segmenter.py` to the specific API.
