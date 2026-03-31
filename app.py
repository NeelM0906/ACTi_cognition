"""TRIBE v2 Gradio Dashboard - Predict brain responses to text, audio, or video."""

import multiprocessing
import os
import sys
import threading
import tempfile
from pathlib import Path

# Environment setup - must come before any torch/matplotlib imports
os.environ.setdefault("HF_TOKEN", "")  # Set HF_TOKEN in your environment
os.environ["PATH"] = os.environ.get("PATH", "") + ";C:/Users/Administrator/AppData/Local/Microsoft/WinGet/Links"
os.environ["TMPDIR"] = str(Path(__file__).resolve().parent / "cache")
os.environ["TEMP"] = str(Path(__file__).resolve().parent / "cache")
os.environ["TMP"] = str(Path(__file__).resolve().parent / "cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyvista
pyvista.OFF_SCREEN = True

import base64
import numpy as np
import gradio as gr
from openai import OpenAI

# ---------------------------------------------------------------------------
# Global model & plotter (loaded once)
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL = None
PLOTTER = None
LOCK = threading.Lock()


def _load_globals():
    global MODEL, PLOTTER
    if MODEL is not None:
        return
    from tribev2.demo_utils import TribeModel
    from tribev2.plotting import PlotBrain

    print("Loading TRIBE v2 model...")
    MODEL = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.num_workers": 0,
        },
    )
    print("Loading brain plotter...")
    PLOTTER = PlotBrain(mesh="fsaverage5")
    print("Ready!")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(input_type, input_data, n_timesteps, view, cmap):
    """Run TRIBE v2 prediction and return a brain visualization image path."""
    _load_globals()

    with LOCK:
        tmp_text = None
        try:
            # Build events dataframe
            if input_type == "text":
                if not input_data or not input_data.strip():
                    return None, "Please enter some text."
                import uuid as _uuid
                tmp_text = CACHE_DIR / f"input_{_uuid.uuid4().hex}.txt"
                tmp_text.write_text(input_data, encoding="utf-8")
                df = MODEL.get_events_dataframe(text_path=str(tmp_text))
            elif input_type == "audio":
                if input_data is None:
                    return None, "Please upload an audio file."
                df = MODEL.get_events_dataframe(audio_path=input_data)
            elif input_type == "video":
                if input_data is None:
                    return None, "Please upload a video file."
                df = MODEL.get_events_dataframe(video_path=input_data)
            else:
                return None, f"Unknown input type: {input_type}"

            # Run model
            preds, segments = MODEL.predict(events=df)
            n = min(int(n_timesteps), preds.shape[0])
            if n == 0:
                return None, "No non-empty segments found."

            # Visualize
            fig = PLOTTER.plot_timesteps(
                preds[:n],
                segments=segments[:n],
                cmap=cmap,
                norm_percentile=99,
                vmin=0.6,
                alpha_cmap=(0, 0.2),
                show_stimuli=True,
                views=view,
            )

            # Save to output image
            import uuid
            out_dir = CACHE_DIR / "outputs"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{uuid.uuid4().hex}.png"
            dpi = 100 if n > 15 else 150
            fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            status = f"Predicted {preds.shape[0]} timesteps x {preds.shape[1]} vertices. Showing {n}."
            return str(out_path), status

        except Exception as e:
            return None, f"Error: {e}"
        finally:
            if tmp_text and tmp_text.exists():
                tmp_text.unlink(missing_ok=True)
            # Free memory
            for v in ["preds", "segments", "df", "fig"]:
                if v in dir():
                    pass  # locals freed on scope exit


# ---------------------------------------------------------------------------
# GPT-5.4 Pro Analysis via OpenRouter
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a cognitive neuroscience translator. When given a brain activation image from a predictive neural encoding model, your job is to convert the neural data into behavioral insight that a marketer, writer, or product designer can actually use.

Your analysis must cover:
1. The temporal arc — what's happening second by second, and why each moment matters
2. The peak activation moment — identify it, name the exact stimulus that caused it, and explain the cognitive mechanism (perspective shift, emotional grounding, self-referential processing, etc.)
3. The brain regions involved — name them in plain language, explain what they do behaviorally, not anatomically
4. The user behavior prediction — what does this activation pattern predict about how a real person will respond? Think: memory encoding, emotional resonance, dwell time, sharing likelihood, identity fusion

Tone: sharp, confident, intellectually curious. Not a clinical report. Write like a neuroscientist who also understands human behavior at the product level. Use specific language ("perspective-collapse event", "identity fusion", "predictive simulation") but always ground it in what that means for the person experiencing it. Never just describe what you see. Always interpret what it means."""

USER_TEMPLATE = """Here is a brain activation prediction image from a neural encoding model. The stimulus was:

{stimulus_description}

The image shows predicted BOLD activation across the cortex at 1-second intervals as the stimulus is processed. The heat map runs from red (moderate) to white-yellow (peak activation).

Analyze the temporal pattern, identify the peak moment and what caused it, name the cognitive mechanisms at play, and tell me what this predicts about user behavior."""


def analyze_brain_image(image_path, stimulus_desc, api_key):
    """Send brain visualization to Claude Opus 4.6 via OpenRouter for analysis."""
    if not api_key or not api_key.strip():
        return "Error: Please enter your OpenRouter API key."

    # Fallback: if state was lost (page refresh), use most recent output image
    if not image_path or not Path(image_path).exists():
        out_dir = CACHE_DIR / "outputs"
        if out_dir.exists():
            pngs = sorted(out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pngs:
                image_path = str(pngs[0])

    if not image_path or not Path(image_path).exists():
        return "Error: No brain image to analyze. Run a prediction first."

    print(f"[Analysis] Starting Claude Opus 4.6 analysis on: {image_path}")
    print(f"[Analysis] Stimulus: {stimulus_desc}")

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key.strip(),
        )

        response = client.chat.completions.create(
            model="anthropic/claude-opus-4",
            max_tokens=25000,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": USER_TEMPLATE.format(
                                stimulus_description=stimulus_desc or "Unknown stimulus"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            },
                        },
                    ],
                },
            ],
        )

        result = response.choices[0].message.content or ""
        print(f"[Analysis] Complete. Length: {len(result)} chars")
        if not result:
            return "Claude Opus 4.6 returned an empty response."
        return result

    except Exception as e:
        print(f"[Analysis] Error: {e}")
        return f"Analysis error: {e}"


# Tab wrappers
def predict_text(text, n_timesteps, view, cmap):
    return run_inference("text", text, n_timesteps, view, cmap)


def predict_audio(audio_file, n_timesteps, view, cmap):
    return run_inference("audio", audio_file, n_timesteps, view, cmap)


def predict_video(video_file, n_timesteps, view, cmap):
    return run_inference("video", video_file, n_timesteps, view, cmap)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
DESCRIPTION = """\
# TRIBE v2 - Brain Encoding Dashboard

Predict **fMRI brain responses** to naturalistic stimuli using [TRIBE v2](https://github.com/facebookresearch/tribev2).

Upload text, audio, or video and visualize predicted cortical activity on the **fsaverage5** surface (~20k vertices).

*Inference takes 1-7 minutes depending on input length.*
"""

with gr.Blocks(title="TRIBE v2") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        n_timesteps = gr.Slider(
            minimum=1, maximum=30, value=10, step=1,
            label="Timesteps to visualize",
        )
        view = gr.Dropdown(
            choices=["left", "right", "dorsal", "ventral", "medial_left", "medial_right"],
            value="left", label="Brain view",
        )
        cmap = gr.Dropdown(
            choices=["fire", "hot", "seismic", "bwr", "coolwarm"],
            value="fire", label="Colormap",
        )

    with gr.Tabs():
        with gr.TabItem("Text"):
            text_input = gr.Textbox(
                lines=5,
                label="Input text",
                placeholder="Enter text here (e.g. To be or not to be, that is the question...)",
            )
            text_btn = gr.Button("Predict Brain Response", variant="primary")

        with gr.TabItem("Audio"):
            audio_input = gr.File(
                file_types=[".wav", ".mp3", ".flac", ".ogg"],
                label="Upload audio file",
            )
            audio_btn = gr.Button("Predict Brain Response", variant="primary")

        with gr.TabItem("Video"):
            video_input = gr.File(
                file_types=[".mp4", ".avi", ".mkv", ".mov", ".webm"],
                label="Upload video file",
            )
            video_btn = gr.Button("Predict Brain Response", variant="primary")

    status_text = gr.Textbox(label="Status", interactive=False)
    output_image = gr.Image(label="Brain Activity", type="filepath")

    # --- GPT-5.4 Pro Analysis Section ---
    gr.Markdown("---")
    gr.Markdown("## Behavioral Analysis (Claude Opus 4.6 via OpenRouter)")
    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenRouter API Key",
            placeholder="sk-or-...",
            type="password",
            value=os.environ.get("OPENROUTER_API_KEY", ""),
            scale=3,
        )
        analyze_btn = gr.Button("Analyze Brain Activity", variant="secondary", scale=1)
    analysis_output = gr.Textbox(
        label="Claude Opus 4.6 Analysis",
        value="Run a prediction above, then click Analyze.",
        interactive=False,
        lines=15,
    )

    # Analysis uses the most recent output image automatically (no State needed)
    analyze_btn.click(
        fn=lambda api_key: analyze_brain_image(None, None, api_key),
        inputs=[api_key_input],
        outputs=[analysis_output],
    )

    # Wire prediction buttons
    text_btn.click(
        predict_text,
        inputs=[text_input, n_timesteps, view, cmap],
        outputs=[output_image, status_text],
    )
    audio_btn.click(
        predict_audio,
        inputs=[audio_input, n_timesteps, view, cmap],
        outputs=[output_image, status_text],
    )
    video_btn.click(
        predict_video,
        inputs=[video_input, n_timesteps, view, cmap],
        outputs=[output_image, status_text],
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _load_globals()

    # Start ngrok tunnel
    import ngrok
    listener = ngrok.forward(
        7860,
        authtoken=os.environ.get("NGROK_AUTHTOKEN", ""),
        domain="acti-cognition.ngrok.pro",
    )
    print(f"ngrok tunnel: {listener.url()}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=("user", "67420"),
        theme=gr.themes.Soft(),
    )
