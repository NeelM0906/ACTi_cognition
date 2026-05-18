"""TRIBE v2 Gradio Dashboard - Predict brain responses to text, audio, or video."""

import multiprocessing
import os
import sys
import threading
import tempfile
from pathlib import Path

# Environment setup - must come before any torch/matplotlib imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")
os.environ.setdefault("HF_TOKEN", "")  # Set HF_TOKEN in your environment
os.environ["TMPDIR"] = str(Path(__file__).resolve().parent / "cache")
os.environ["TEMP"] = str(Path(__file__).resolve().parent / "cache")
os.environ["TMP"] = str(Path(__file__).resolve().parent / "cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyvista
pyvista.OFF_SCREEN = True

import base64
import secrets
from typing import Literal

import numpy as np
import gradio as gr
from openai import OpenAI
from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

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


def analyze_brain_image(image_path, stimulus_desc, api_key, model=None):
    """Send brain visualization to a vision-LLM via OpenRouter for analysis."""
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

    resolved_model = model or os.environ.get("OPENROUTER_ANALYSIS_MODEL", "anthropic/claude-opus-4")
    print(f"[Analysis] Starting {resolved_model} analysis on: {image_path}")
    print(f"[Analysis] Stimulus: {stimulus_desc}")

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key.strip(),
        )

        response = client.chat.completions.create(
            model=resolved_model,
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
            return f"{resolved_model} returned an empty response."
        return result

    except Exception as e:
        print(f"[Analysis] Error: {e}")
        return f"Analysis error: {e}"


# ---------------------------------------------------------------------------
# External-dev unified endpoint: text -> brain prediction -> LLM analysis
# Returns ONLY the analysis text. Uses server-side OPENROUTER_API_KEY.
# ---------------------------------------------------------------------------
def analyze_text_endpoint(text, n_timesteps=10, view="left", cmap="fire"):
    """Run TRIBE on text input and return only the LLM analysis of the brain image."""
    if not text or not str(text).strip():
        return "Error: Please provide non-empty input text."

    server_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not server_key:
        return "Error: Server is missing OPENROUTER_API_KEY in .env."

    image_path, status = run_inference("text", text, n_timesteps, view, cmap)
    if not image_path:
        return f"Error: brain prediction failed: {status}"

    return analyze_brain_image(
        image_path=image_path,
        stimulus_desc=text,
        api_key=server_key,
        model=os.environ.get("OPENROUTER_ANALYSIS_MODEL", "xiaomi/mimo-v2.5-pro"),
    )


# ---------------------------------------------------------------------------
# Media (audio / video) -> brain prediction -> LLM analysis.
# Mirrors analyze_text_endpoint but takes a file path produced from an upload.
# ---------------------------------------------------------------------------
def _analyze_media_endpoint(input_type, file_path, n_timesteps, view, cmap):
    server_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not server_key:
        return "Error: Server is missing OPENROUTER_API_KEY in .env."

    image_path, info = run_inference(input_type, str(file_path), n_timesteps, view, cmap)
    if not image_path:
        return f"Error: brain prediction failed: {info}"

    return analyze_brain_image(
        image_path=image_path,
        stimulus_desc=f"{input_type.capitalize()} stimulus: {Path(file_path).name}",
        api_key=server_key,
        model=os.environ.get("OPENROUTER_ANALYSIS_MODEL", "xiaomi/mimo-v2.5-pro"),
    )


# ---------------------------------------------------------------------------
# Clean REST API surface for external dev / agent integration.
# A thin FastAPI layer in front of the existing Gradio app.
# ---------------------------------------------------------------------------
_security = HTTPBasic()


def _check_auth(creds: HTTPBasicCredentials = Depends(_security)):
    ok_user = secrets.compare_digest(creds.username, "user")
    ok_pass = secrets.compare_digest(creds.password, "67420")
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Stimulus text. The brain response is predicted for this text, then analyzed.")
    n_timesteps: int = Field(10, ge=1, le=30, description="How many predicted brain-activation timesteps to render (1-30).")
    view: Literal["left", "right", "dorsal", "ventral", "medial_left", "medial_right"] = Field("left", description="Cortical surface view.")
    cmap: Literal["fire", "hot", "seismic", "bwr", "coolwarm"] = Field("fire", description="Colormap for the activation heatmap.")


class AnalysisResponse(BaseModel):
    analysis: str = Field(..., description="Natural-language behavioral interpretation of the predicted brain response.")


_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
_VALID_VIEWS = {"left", "right", "dorsal", "ventral", "medial_left", "medial_right"}
_VALID_CMAPS = {"fire", "hot", "seismic", "bwr", "coolwarm"}


def _save_upload(upload: UploadFile, allowed_suffixes: set[str], kind: str) -> Path:
    """Stream an UploadFile to CACHE_DIR/uploads with a random name + validated suffix."""
    import uuid as _uuid
    src_name = upload.filename or ""
    suffix = Path(src_name).suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{kind} file must end with one of {sorted(allowed_suffixes)}, "
                f"got '{suffix}' (filename='{src_name}')"
            ),
        )
    uploads_dir = CACHE_DIR / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    dst = uploads_dir / f"{kind}_{_uuid.uuid4().hex}{suffix}"
    with dst.open("wb") as f:
        while True:
            chunk = upload.file.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            f.write(chunk)
    return dst


def _validate_render_params(n_timesteps: int, view: str, cmap: str) -> None:
    if not (1 <= n_timesteps <= 30):
        raise HTTPException(status_code=422, detail=f"n_timesteps must be in [1, 30], got {n_timesteps}")
    if view not in _VALID_VIEWS:
        raise HTTPException(status_code=422, detail=f"invalid view '{view}'; allowed: {sorted(_VALID_VIEWS)}")
    if cmap not in _VALID_CMAPS:
        raise HTTPException(status_code=422, detail=f"invalid cmap '{cmap}'; allowed: {sorted(_VALID_CMAPS)}")


def build_api(gradio_demo):
    app = FastAPI(
        title="TRIBE Brain Analysis API",
        version="1.0.0",
        description="Predict brain response to a stimulus and return a behavioral interpretation.",
    )

    @app.post(
        "/analysis",
        response_model=AnalysisResponse,
        dependencies=[Depends(_check_auth)],
        summary="Predict brain response to text and return behavioral analysis",
    )
    def analysis(req: AnalysisRequest):
        result = analyze_text_endpoint(req.text, req.n_timesteps, req.view, req.cmap)
        if result.startswith("Error:") or result.startswith("Analysis error:"):
            raise HTTPException(status_code=502, detail=result)
        return AnalysisResponse(analysis=result)

    @app.post(
        "/analysis/audio",
        response_model=AnalysisResponse,
        dependencies=[Depends(_check_auth)],
        summary="Predict brain response to uploaded audio and return behavioral analysis",
    )
    def analysis_audio(
        file: UploadFile = File(..., description="Audio file: .wav, .mp3, .flac, or .ogg"),
        n_timesteps: int = Form(10),
        view: str = Form("left"),
        cmap: str = Form("fire"),
    ):
        _validate_render_params(n_timesteps, view, cmap)
        saved = _save_upload(file, _AUDIO_SUFFIXES, "audio")
        try:
            result = _analyze_media_endpoint("audio", saved, n_timesteps, view, cmap)
        finally:
            try:
                saved.unlink()
            except OSError:
                pass
        if result.startswith("Error:") or result.startswith("Analysis error:"):
            raise HTTPException(status_code=502, detail=result)
        return AnalysisResponse(analysis=result)

    @app.post(
        "/analysis/video",
        response_model=AnalysisResponse,
        dependencies=[Depends(_check_auth)],
        summary="Predict brain response to uploaded video and return behavioral analysis",
    )
    def analysis_video(
        file: UploadFile = File(..., description="Video file: .mp4, .avi, .mkv, .mov, or .webm"),
        n_timesteps: int = Form(10),
        view: str = Form("left"),
        cmap: str = Form("fire"),
    ):
        _validate_render_params(n_timesteps, view, cmap)
        saved = _save_upload(file, _VIDEO_SUFFIXES, "video")
        try:
            result = _analyze_media_endpoint("video", saved, n_timesteps, view, cmap)
        finally:
            try:
                saved.unlink()
            except OSError:
                pass
        if result.startswith("Error:") or result.startswith("Analysis error:"):
            raise HTTPException(status_code=502, detail=result)
        return AnalysisResponse(analysis=result)

    @app.get("/healthz", summary="Liveness check")
    def healthz():
        return {"status": "ok", "model_loaded": MODEL is not None}

    return gr.mount_gradio_app(app, gradio_demo, path="/", auth=("user", "67420"))


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

with gr.Blocks(title="TRIBE v2", theme=gr.themes.Soft()) as demo:
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

    # --- Behavioral Analysis Section ---
    gr.Markdown("---")
    gr.Markdown(
        "## Behavioral Analysis\n"
        "Each tab's **Predict Brain Response** button automatically runs the analysis after "
        "prediction. Leave the API key empty to use the server's configured OpenRouter key."
    )
    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenRouter API Key (optional — server key used if empty)",
            placeholder="sk-or-...  (leave blank to use server key)",
            type="password",
            value="",
            scale=3,
        )
        analyze_btn = gr.Button("Re-analyze last brain image", variant="secondary", scale=1)
    analysis_output = gr.Textbox(
        label="Behavioral analysis (Markdown)",
        value="Upload a stimulus above and click Predict — the analysis will appear here.",
        interactive=False,
        lines=15,
    )

    def _gradio_analyze_with_fallback(api_key, stimulus_desc=None):
        key = (api_key or "").strip() or os.environ.get("OPENROUTER_API_KEY", "")
        return analyze_brain_image(None, stimulus_desc, key)

    # "Re-analyze" button — uses the most recent output image with the chosen API key.
    analyze_btn.click(
        fn=_gradio_analyze_with_fallback,
        inputs=[api_key_input],
        outputs=[analysis_output],
    )

    # --- Hidden API-only endpoint for external devs ---
    # Single call: text -> TRIBE prediction -> LLM analysis -> return analysis string.
    # Uses the server's OPENROUTER_API_KEY from .env; no auth-related params from caller.
    _api_text = gr.Textbox(visible=False)
    _api_n = gr.Number(value=10, visible=False)
    _api_view = gr.Textbox(value="left", visible=False)
    _api_cmap = gr.Textbox(value="fire", visible=False)
    _api_out = gr.Textbox(visible=False)
    _api_btn = gr.Button(visible=False)
    _api_btn.click(
        fn=analyze_text_endpoint,
        inputs=[_api_text, _api_n, _api_view, _api_cmap],
        outputs=[_api_out],
        api_name="analyze_text",
    )

    # Wire prediction buttons — each chains into LLM analysis automatically.
    # The analysis step picks up the most recent output PNG via analyze_brain_image's
    # fallback, and uses the server's OPENROUTER_API_KEY when the textbox is empty.
    text_btn.click(
        predict_text,
        inputs=[text_input, n_timesteps, view, cmap],
        outputs=[output_image, status_text],
        api_name="predict_text",
    ).then(
        fn=lambda api_key, stim: _gradio_analyze_with_fallback(api_key, stim or "text stimulus"),
        inputs=[api_key_input, text_input],
        outputs=[analysis_output],
    )
    audio_btn.click(
        predict_audio,
        inputs=[audio_input, n_timesteps, view, cmap],
        outputs=[output_image, status_text],
    ).then(
        fn=lambda api_key, f: _gradio_analyze_with_fallback(
            api_key, f"Audio stimulus: {Path(f).name}" if f else "audio stimulus"
        ),
        inputs=[api_key_input, audio_input],
        outputs=[analysis_output],
    )
    video_btn.click(
        predict_video,
        inputs=[video_input, n_timesteps, view, cmap],
        outputs=[output_image, status_text],
    ).then(
        fn=lambda api_key, f: _gradio_analyze_with_fallback(
            api_key, f"Video stimulus: {Path(f).name}" if f else "video stimulus"
        ),
        inputs=[api_key_input, video_input],
        outputs=[analysis_output],
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _load_globals()

    # Build the combined FastAPI app: REST API at the root + Gradio UI mounted under it.
    api_app = build_api(demo)

    # Start ngrok tunnel
    import ngrok
    listener = ngrok.forward(
        7860,
        authtoken=os.environ.get("NGROK_AUTHTOKEN", ""),
        domain=os.environ.get("NGROK_DOMAIN", "acti.cognition.ngrok.pro"),
    )
    print(f"ngrok tunnel: {listener.url()}")
    print(f"REST API:    {listener.url()}/analysis")
    print(f"OpenAPI:     {listener.url()}/docs")

    import uvicorn
    uvicorn.run(api_app, host="0.0.0.0", port=7860, log_level="info")
