"""Quick test of TRIBE v2: load model, predict brain response to text."""
import os
os.environ.setdefault("HF_TOKEN", "")  # Set HF_TOKEN in your environment

import multiprocessing
from pathlib import Path
from tribev2.demo_utils import TribeModel

CACHE_FOLDER = Path("./cache")


def main():
    print("=== Step 1: Loading TRIBE v2 model from HuggingFace ===")
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_FOLDER,
        config_update={"data.text_feature.model_name": "unsloth/Llama-3.2-3B"},
    )
    print("Model loaded successfully!\n")

    # Test with text input (simplest - no video/audio file needed)
    print("=== Step 2: Preparing text input ===")
    text = """
To be or not to be, that is the question.
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune.
"""

    text_path = CACHE_FOLDER / "test_text.txt"
    text_path.write_text(text)

    df = model.get_events_dataframe(text_path=text_path)
    print(f"Events dataframe shape: {df.shape}")
    print(df.head(8)[["type", "start", "duration", "text"]].to_string())
    print()

    print("=== Step 3: Running model prediction ===")
    preds, segments = model.predict(events=df)
    print(f"\nPredictions shape: {preds.shape}  (n_timesteps, n_vertices)")
    print(f"Number of segments: {len(segments)}")
    print("\nDone! TRIBE v2 is working correctly.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
