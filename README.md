# DDS-Hoax-Probability
Tools and utilities for extracting incident data and estimating hoax probability / severity.

This repository contains small, focused modules used to process incident reports and evidence, extract structured information, and provide heuristic scores for hoax-likelihood and incident urgency/severity. The code is intentionally modular so you can pick the pieces you need (offline/local heuristics + optional LLM/vision integrations).

## Features

- Image metadata extraction
	- `get_img_metadata.py` reads EXIF, PNG ancillary chunks, GPS, and Apple-specific hints (HEIC/ProRAW, Live Photo pairing).
- Similarity-based hoax scoring
	- `get_similarity_hoax.py` embeds report text with a SentenceTransformers model and computes cluster stats and a tunable `hoax_score` based on anonymity, template-like similarity, and named corroboration.
- Incident extraction via vision-enabled LLM
	- `gpt_incident_agent.py` is a FastAPI service that accepts `report_id`, `report` text and attachments (images/PDFs). It sends mixed text+images to an OpenAI vision-capable model and requests strict JSON extraction matching a defined schema.
- OCR utilities
	- `ocr.py` demonstrates extracting text from images using an OpenAI chat/vision model. (There is also `old_ocr.py` preserved for legacy approaches.)
- Deterministic incident classifier and urgency scoring
	- `incident_severity_score.py` contains a regex-based incident classifier, severity mapping, urgency heuristic, and wrappers for optional LLM fallbacks (OpenAI, Hugging Face, or local Transformers).

## Quickstart

1) Create and activate a Python environment. Example (Windows PowerShell):

	 ```powershell
	 python -m venv .venv; . .\.venv\Scripts\Activate.ps1
	 pip install -r requirements.txt  # if you create one, or install per-module deps below
	 ```

2) Minimal per-module dependencies (install only what you need):

	 - `get_img_metadata.py`: Pillow (optional `pillow-heif` for HEIC)
	 - `get_similarity_hoax.py`: sentence-transformers, numpy
	 - `gpt_incident_agent.py`: fastapi, uvicorn, openai, python-multipart, pillow, pdf2image, pytesseract (OCR optional)
	 - `ocr.py`: openai
	 - `incident_severity_score.py`: no network required for the local classifier; optional `openai`, `huggingface_hub`, `transformers`, `accelerate` for LLM fallbacks

	 Example:

	 ```powershell
	 pip install pillow sentence-transformers numpy fastapi uvicorn openai python-multipart pdf2image pytesseract
	 ```

3) Environment variables

	 - `OPENAI_API_KEY` — required for OpenAI-based features (vision, OCR, LLM fallbacks)
	 - `HF_API_TOKEN` or `HUGGINGFACE_API_TOKEN` — optional, for Hugging Face Inference API
	 - `LOCAL_TRANSFORMERS_MODEL` — optional, local model id if running offline Transformers
	 - `SOC_SALT` — optional salt used to hash subject-of-concern (SoC) keys

	 Important: Do NOT commit secrets. Example run scripts in this repo (`run_incident_severity.ps1` / `.sh`) include placeholder/example env assignments; remove any keys before committing.

## Usage examples

- Read image metadata (Python):

	```py
	from get_img_metadata import read_image_metadata
	meta = read_image_metadata('images/example.jpg')
	print(meta['gps'], meta['exif'].get('Make'))
	```

- Compute similarity + hoax score (Python):

	```py
	from get_similarity_hoax import score_similarity_and_hoax
	new = {'report_id':'NEW1','text':'People seen with guns near the east gate.'}
	candidates = [{'report_id':'A','text':'Shots heard by east gate','is_anonymous':True}, ...]
	res = score_similarity_and_hoax(new, candidates)
	print(res['summary'], res['hoax_score'])
	```

- Run the FastAPI incident extractor:

	```powershell
	$env:OPENAI_API_KEY = '<your key>'
	uvicorn gpt_incident_agent:app --reload --port 8000
	```

	POST `/v1/extract` accepts `report_id`, `report` (form fields) and file uploads. It returns a strict JSON extraction matching the schema defined in the code.

- Run the local deterministic classifier demo:

	```powershell
	python incident_severity_score.py
	```


