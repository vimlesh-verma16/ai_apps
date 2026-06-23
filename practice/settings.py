import os
import sys
from pathlib import Path

def load_env_file() -> None:
    """Load variables from a local .env file if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file()
MODEL_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Add a check for clarity
if MODEL_API_KEY is None:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.")