# ocr.py
import os
import base64
import mimetypes
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv  # pip install python-dotenv
from openai import OpenAI

# Load environment variables from a local .env file if present (not committed)
load_dotenv()

# Never hardcode secrets. Read from env instead.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Put it in your environment or a local .env file (which is gitignored)."
    )

# Initialize the client with the key from env
client = OpenAI(api_key=OPENAI_API_KEY)
HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "images"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def to_data_url(p: Path) -> str:
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def ocr_with_gpt(p: Path) -> str:
    data_url = to_data_url(p)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # "gpt-4o" is a bit stronger but pricier
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract ONLY the raw text. No commentary."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ],
        }],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def main():
    if not IMG_DIR.exists():
        print(f"[ERROR] Folder not found: {IMG_DIR}")
        return
    imgs = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS and p.is_file()])
    if not imgs:
        print(f"No images found in {IMG_DIR}")
        return

    for p in imgs:
        try:
            text = ocr_with_gpt(p)
        except Exception as e:
            text = f"[ERROR: {e}]"
        print(f"{p.name}: {text}")

if __name__ == "__main__":
    main()
