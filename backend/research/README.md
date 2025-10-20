## Textbook Problem Detection

This repository extracts exercise/problem blocks from textbook pages by
combining OCR output with the OpenAI multimodal API.

### Requirements

- Python 3.12+
- [`httpx`](https://www.python-httpx.org/) for async OpenAI access
- Optional:
  - [`pdf2image`](https://pypi.org/project/pdf2image/) + Poppler or the `pdftoppm`
    CLI for PDF rendering.
  - [`pillow`](https://pypi.org/project/pillow/) for generating annotated preview
    images.

Set `OPENAI_API_KEY` in your environment to avoid hard-coding credentials:

```bash
export OPENAI_API_KEY="sk-..."
```

### Usage

```bash
python main.py \
  --pdf data/math.pdf \
  --ocr ocr_result_pass.json \
  --output-dir run_artifacts
```

This will:

1. Render every PDF page to `run_artifacts/page_images/`.
2. Send each page image + OCR lines to the OpenAI vision model and collect the
   list of problems with the corresponding OCR line indices.
3. Save the structured result to
   `run_artifacts/outputs/problem_assignments.json` (includes the full problem
   text plus a per-line validation flag showing whether each OCR line is part of
   that problem).
4. Persist the raw model responses for each page in
   `run_artifacts/outputs/llm_raw/`.
5. (Optional) Draw bounding boxes labelled with the problem identifier for
   visual verification (`run_artifacts/outputs/visualisations/`). Lines marked as
   mismatches are highlighted in red with a trailing `!` to speed up review.

Disable step 5 with `--no-visualise`.

### Configuration

Key runtime parameters can be tweaked via environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `OPENAI_VISION_MODEL` | Model used for extraction | `gpt-4o-mini` |
| `TEXTBOOK_PDF_DPI` | DPI for PDF rendering | `220` |
| `TEXTBOOK_MAX_OCR_LINES` | Max OCR lines passed to the model (per page) | `160` |
| `LLM_MAX_CONCURRENCY` | Parallel OpenAI requests when processing pages | `5` |

See `detector/config.py` for the full list.
