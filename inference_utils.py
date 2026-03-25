import json
import os

THRESHOLD_FILE = "best_threshold.json"


def load_best_threshold(default=0.75):
    if not os.path.exists(THRESHOLD_FILE):
        return default

    try:
        with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("threshold", default))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return default
