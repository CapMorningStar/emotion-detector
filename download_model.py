"""
Downloads pretrained mini_XCEPTION weights trained on FER-2013.
Source: https://github.com/oarriaga/face_classification
"""
import os
import requests

MODEL_DIR = os.path.join(os.path.dirname(__file__), "weights")
MODEL_URL = (
    "https://github.com/oarriaga/face_classification/raw/master/"
    "trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
)
MODEL_PATH = os.path.join(MODEL_DIR, "fer2013_mini_XCEPTION.hdf5")


def download_weights():
    if os.path.exists(MODEL_PATH):
        print(f"Weights already exist at {MODEL_PATH}")
        return MODEL_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading pretrained weights...")

    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%", end="", flush=True)

    print(f"\nSaved to {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    download_weights()
