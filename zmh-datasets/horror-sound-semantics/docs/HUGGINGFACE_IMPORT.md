# Hugging Face datasets: discover, preview, and import safely

This small tool helps you investigate public audio datasets on the Hugging Face Hub. It uses the Hub metadata API and the datasets-server row viewer. **Discovery and preview do not clone repositories or download full datasets.** Preview shows at most five rows. Imports save only those rows' metadata to `data/third_party/huggingface/`; audio bytes are never downloaded.

## Install

From this folder (`zmh-datasets/horror-sound-semantics/`), install the project's lightweight requirements. `requests` is already included:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Discover candidates

```bash
python -m src.huggingface_integration discover --limit 20
```

Search covers horror sound effects, sound events, cinematic audio, and foley. Output includes dataset ID, license tag, reported size, available splits, description, source URL, and a conservative license-screening note. CC0 and CC BY 4.0 candidates sort first. When reachable, the tool also inspects the public dataset README/card text for restrictions; network/readme failures are recorded as absent evidence, not as approval. The Hub's license tag and card are only leads: this tool always says manual review is required.

## 2. Preview five rows

```bash
python -m src.huggingface_integration preview some-org/some-audio-dataset
python -m src.huggingface_integration preview some-org/some-audio-dataset --split train
```

The request is capped at five rows through the Hub datasets-server API. Audio cells may contain URLs or references; the preview does not fetch the audio files. Some datasets are not supported by the viewer and may return an API error; try another split/configuration or review the dataset repository directly.

## 3. Review rights before importing

Before importing, open the `source_url` and check the dataset card, linked terms, and—especially for audio—the rights to the underlying recordings. Confirm the stated license covers redistribution of the actual media, not just annotations or code. Check attribution wording, third-party content, privacy/personality rights, and any additional restrictions. If rights are unclear, **do not import**. The automated license screen is not legal advice and cannot establish rights.

The importer accepts only Hub tags `cc0-1.0` and `cc-by-4.0`, requires explicit acknowledgement, and stores a small metadata manifest separately from ZMH-authored audio:

```bash
# Five metadata records by default; no audio files downloaded
python -m src.huggingface_integration import some-org/some-audio-dataset \
  --split train --limit 5 --approve
```

`--approve` confirms that you performed your review; it does not bypass the license allowlist or claim that rights were independently verified. Import is capped at 100 metadata rows per run. No dataset is published, and no licensing terms are changed.

## Provenance and schema behavior

Each imported entry records its Hub dataset ID, row ID, dataset revision (commit hash where available), original license, source URL, CC BY attribution indication/text, and the original row. `zmh_schema_mapping` is a **partial** mapping of recognizable source columns (audio, caption, event label, temporal events) to corresponding ZMH concepts. It does not make the record a valid ZMH training record: missing values stay missing, emotion/tension judgments are not fabricated, and imported records are not placed in `data/demo_audio/` or the original `training_data.jsonl`. Human review remains `unreviewed_provisional`; annotation remains `not_annotated` until reviewers actually annotate it. This deliberate separation preserves the source's identity and avoids implying that third-party recordings were created by ZMH.

The output file is deterministic per dataset ID and duplicate source record IDs are ignored on repeat imports. Keep the import under `data/third_party/huggingface/` so it remains visibly distinct from original audio.

## Troubleshooting

- `401`/`403`: the dataset may be gated or private; this CLI currently supports public datasets only. Do not add a token to source code.
- No preview rows: check that the dataset has a supported viewer config and the requested split exists.
- License rejected: inspect the original terms manually. The tool intentionally does not import other license tags.
- Network unavailable: discovery/preview require access to Hugging Face Hub and datasets-server APIs.

These commands are discovery and local metadata tooling only. They do not grant redistribution rights or authorize publication, and all candidate datasets require a human approval decision.
