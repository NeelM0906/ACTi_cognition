"""
app.py — A/B Test Simulator powered by Meta's TRIBE v2 brain-response model.

Supports per-variant modality modes: Video, Audio, Text, Image (static→video), and Multimodal
(video + audio + text ± image). Results show aligned time courses, ROI systems, and modality notes.

Run:
    python app.py [--share] [--port 7860]
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from typing import Optional

import gradio as gr

from tribe_utils import (
    ABComparator,
    BrainVisualizer,
    ModalityMode,
    StimulusSpec,
    TribeInferenceEngine,
    build_summary_dataframe,
    effective_stimulus_spec,
    format_winner_markdown,
    validate_stimulus_spec,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

ALL_MODALITY_CHOICES = [
    ("Text", ModalityMode.TEXT.value),
    ("Image", ModalityMode.IMAGE.value),
    ("Audio", ModalityMode.AUDIO.value),
    ("Video", ModalityMode.VIDEO.value),
]

engine = TribeInferenceEngine(cache_dir="./cache", device="auto")

# ---------------------------------------------------------------------------
# Presets — text-focused examples (binary assets not bundled)
# ---------------------------------------------------------------------------

PRESETS = {
    "Ad copy: Emotional vs Rational (text)": {
        "mode_a": ModalityMode.TEXT.value,
        "mode_b": ModalityMode.TEXT.value,
        "text_a": (
            "LIMITED TIME: Transform your mornings in just 5 minutes a day. "
            "Join 2 million people who start calmer, sharper, more energized. "
            "Tap below — your future self will thank you."
        ),
        "text_b": (
            "Our morning routine app is used by 2 million people. "
            "5-minute daily sessions. Evidence-based techniques. Download now."
        ),
        "name_a": "Emotional Hook",
        "name_b": "Rational CTA",
        "hint": "",
    },
    "Educational: Story vs Facts (text)": {
        "mode_a": ModalityMode.TEXT.value,
        "mode_b": ModalityMode.TEXT.value,
        "text_a": (
            "Maria always struggled with math until the day her teacher showed her "
            "how numbers are everywhere — in music, in nature, in the stars."
        ),
        "text_b": (
            "Studies show that contextual learning improves retention by 42%. "
            "Spaced repetition reduces forgetting by 80% over 30 days."
        ),
        "name_a": "Story-Based",
        "name_b": "Fact-Based",
        "hint": "",
    },
    "Social: Inspiring vs Neutral (text)": {
        "mode_a": ModalityMode.TEXT.value,
        "mode_b": ModalityMode.TEXT.value,
        "text_a": (
            "You have the power to change the world. Every action, no matter how small, "
            "creates ripples that reach far beyond what you can see."
        ),
        "text_b": (
            "This post contains information about community events happening in your area "
            "this month. Click to see the full schedule."
        ),
        "name_a": "Inspiring Post",
        "name_b": "Neutral Post",
        "hint": "",
    },
    "Image A vs Image B": {
        "mode_a": ModalityMode.IMAGE.value,
        "mode_b": ModalityMode.IMAGE.value,
        "text_a": "",
        "text_b": "",
        "name_a": "Image A",
        "name_b": "Image B",
        "hint": "Upload two image variants to compare neural response.",
    },
}

PRESET_NAMES = ["(none)"] + list(PRESETS.keys())


# ---------------------------------------------------------------------------
# Visibility toggles for modality groups
# ---------------------------------------------------------------------------


def _toggle_input_groups(mode: str):
    m = (mode or ModalityMode.TEXT.value).lower()
    return (
        gr.update(visible=m == ModalityMode.TEXT.value),
        gr.update(visible=m == ModalityMode.IMAGE.value),
        gr.update(visible=m == ModalityMode.AUDIO.value),
        gr.update(visible=m == ModalityMode.VIDEO.value),
    )


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------


def _snippet(text: str, n: int = 400) -> str:
    t = (text or "").strip()
    if len(t) <= n:
        return t
    return t[:n] + "…"


def build_input_preview_md(label: str, spec: StimulusSpec) -> str:
    """Human-readable checklist (matches `effective_stimulus_spec`, no stale fields)."""
    m = (spec.mode or "").strip()
    lines = [f"### {label}", f"**Mode:** `{m}`  "]
    if spec.video_path:
        lines.append("- Video file attached ✓")
    if spec.audio_path:
        lines.append("- Audio file attached ✓")
    if (spec.text or "").strip():
        lines.append(f"- **Text:** “{_snippet(spec.text or '')}”")
    if spec.image_path:
        lines.append("- Image file attached ✓ (for Image mode or multimodal visual)")
    if m == ModalityMode.TEXT.value and not (spec.text or "").strip():
        lines.append("- ⚠️ _No text yet_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A/B pipeline
# ---------------------------------------------------------------------------


def run_ab_test(
    mode_a: str,
    name_a: str,
    text_a: str,
    image_a: Optional[str],
    audio_a: Optional[str],
    video_a: Optional[str],
    mode_b: str,
    name_b: str,
    text_b: str,
    image_b: Optional[str],
    audio_b: Optional[str],
    video_b: Optional[str],
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    spec_a = effective_stimulus_spec(mode_a, video_a, audio_a, text_a, image_a)
    spec_b = effective_stimulus_spec(mode_b, video_b, audio_b, text_b, image_b)
    err_a = validate_stimulus_spec(spec_a)
    err_b = validate_stimulus_spec(spec_b)
    prev_a = build_input_preview_md("Variant A — stimulus check", spec_a)
    prev_b = build_input_preview_md("Variant B — stimulus check", spec_b)

    if err_a or err_b:
        msg = "\n".join(filter(None, [err_a, err_b]))
        return (
            gr.update(value=f"❌ {msg}", visible=True),
            gr.update(value=prev_a),
            gr.update(value=prev_b),
            None, None, None, None,
            None, None, None,
            None, None,
            None,
            gr.update(visible=False),
        )

    la = name_a.strip() or "A"
    lb = name_b.strip() or "B"
    demo_banner = ""

    try:
        progress(0.08, desc=f"TRIBE inference: {la} …")
        result_a = engine.predict(la, spec_a)

        progress(0.42, desc=f"TRIBE inference: {lb} …")
        result_b = engine.predict(lb, spec_b)

        if engine.is_demo_mode:
            demo_banner = (
                "\n\n> ⚠️ **Demo Mode** — TRIBE v2 not loaded. "
                "Predictions are synthetic for UI testing."
            )

        progress(0.62, desc="Comparing variants (aligned time + stats)…")
        comparison = ABComparator.compare(result_a, result_b)

        progress(0.72, desc="Building charts…")
        fig_timeline = BrainVisualizer.activation_over_time(comparison)
        fig_regions = BrainVisualizer.region_comparison_bar(comparison)
        fig_roi = BrainVisualizer.roi_system_bar(comparison)
        fig_radar = BrainVisualizer.proxy_score_radar(comparison)

        progress(0.85, desc="Rendering brain surfaces…")
        import io as _io
        import matplotlib.pyplot as _plt
        from PIL import Image as _PILImage

        def _fig_to_pil(fig, facecolor="#ffffff") -> "_PILImage.Image":
            buf = _io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor=facecolor)
            _plt.close(fig)
            buf.seek(0)
            return _PILImage.open(buf).copy()

        # In demo mode skip the nilearn ~80MB mesh download (hangs on HF Space startup).
        if engine.is_demo_mode:
            brain_a_bytes = _fig_to_pil(
                BrainVisualizer._fallback_activation_map(
                    result_a.preds.max(axis=0), title=f"Max activation — {la}"
                )
            )
            brain_b_bytes = _fig_to_pil(
                BrainVisualizer._fallback_activation_map(
                    result_b.preds.max(axis=0), title=f"Max activation — {lb}"
                )
            )
        else:
            raw_a = BrainVisualizer.brain_surface_plot(result_a.preds, title=f"Max activation — {la}")
            raw_b = BrainVisualizer.brain_surface_plot(result_b.preds, title=f"Max activation — {lb}")
            brain_a_bytes = _PILImage.open(_io.BytesIO(raw_a)).copy()
            brain_b_bytes = _PILImage.open(_io.BytesIO(raw_b)).copy()

        progress(0.92, desc="Exporting CSV…")
        summary_df = build_summary_dataframe(comparison)
        csv_path = tempfile.NamedTemporaryFile(
            suffix="_ab_results.csv", delete=False, mode="w"
        ).name
        summary_df.to_csv(csv_path, index=False)

        winner_md = format_winner_markdown(comparison) + demo_banner

        prev_a2 = prev_a + f"\n\n**Ran as:** `{', '.join(result_a.modalities)}` · **Duration** {result_a.duration_sec:.2f}s"
        prev_b2 = prev_b + f"\n\n**Ran as:** `{', '.join(result_b.modalities)}` · **Duration** {result_b.duration_sec:.2f}s"

        progress(1.0, desc="Done.")

        return (
            gr.update(value=winner_md, visible=True),
            gr.update(value=prev_a2),
            gr.update(value=prev_b2),
            fig_timeline,
            fig_regions,
            fig_roi,
            fig_radar,
            _gauge_figure(comparison.engagement_a, comparison.engagement_b, "Engagement"),
            _gauge_figure(comparison.attention_a, comparison.attention_b, "Attention"),
            _gauge_figure(comparison.emotion_a, comparison.emotion_b, "Emotion"),
            brain_a_bytes,
            brain_b_bytes,
            csv_path,
            gr.update(visible=True),
        )
    except Exception as exc:
        logger.exception("A/B run failed")
        return (
            gr.update(value=f"❌ Error: {exc}", visible=True),
            gr.update(value=prev_a),
            gr.update(value=prev_b),
            None, None, None, None,
            None, None, None,
            None, None,
            None,
            gr.update(visible=False),
        )



def _gauge_figure(val_a: float, val_b: float, title: str):
    import plotly.graph_objects as go

    fig = go.Figure()
    for val, color, name in [
        (val_a, "#60a5fa", "A"),
        (val_b, "#f97316", "B"),
    ]:
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                title={
                    "text": f"{title}<br><span style='color:{color}'>Variant {name}</span>",
                    "font": {"size": 13},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6b9fc4"},
                    "bar": {"color": color},
                    "bgcolor": "rgba(230,245,255,0.5)",
                    "bordercolor": "rgba(100,160,210,0.25)",
                    "steps": [
                        {"range": [0, 33],  "color": "rgba(220,240,255,0.4)"},
                        {"range": [33, 66], "color": "rgba(190,225,252,0.5)"},
                        {"range": [66, 100],"color": "rgba(150,205,245,0.6)"},
                    ],
                    "threshold": {
                        "line": {"color": "#0a2a40", "width": 2},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
                domain={"row": 0 if name == "A" else 1, "column": 0},
            )
        )

    fig.update_layout(
        grid={"rows": 2, "columns": 1, "pattern": "independent"},
        paper_bgcolor="#ffffff",
        font=dict(color="#0a2a40", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        modebar_remove=["pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
                        "autoScale2d","hoverClosestCartesian",
                        "hoverCompareCartesian","toggleSpikelines","toImage",
                        "sendDataToCloud","editInChartStudio"],
    )
    return fig


def load_preset(preset_name: str):
    if preset_name == "(none)" or not preset_name:
        return (
            gr.update(), gr.update(),
            gr.update(value=""), gr.update(value=""),
            gr.update(value="Variant A"), gr.update(value="Variant B"),
            gr.update(value=""),
        )
    p = PRESETS[preset_name]
    return (
        gr.update(value=p.get("mode_a", ModalityMode.TEXT.value)),
        gr.update(value=p.get("mode_b", ModalityMode.TEXT.value)),
        gr.update(value=p.get("text_a", "")),
        gr.update(value=p.get("text_b", "")),
        gr.update(value=p.get("name_a", "A")),
        gr.update(value=p.get("name_b", "B")),
        gr.update(value=f"_{p.get('hint', '')}_" if p.get("hint") else ""),
    )


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

/* ── CSS variable overrides ── */
:root {
    --block-label-background-fill: transparent !important;
    --block-label-text-color: rgba(30,80,140,0.6) !important;
    --block-label-border-color: transparent !important;
    --block-label-padding: 0 !important;
}

/* Force body-text CSS variable dark on every element so nothing inherits a light value */
*, *::before, *::after {
    --body-text-color: #0a2a40 !important;
    --body-text-color-subdued: rgba(10,42,64,0.75) !important;
}

/* ── Ocean-pearl page background ── */
body, .gradio-container {
    background: #dff0f8 !important;
    background-image:
        radial-gradient(ellipse 110% 60% at 50% -10%, rgba(255,240,210,0.75) 0%, rgba(255,220,180,0.3) 35%, transparent 65%),
        radial-gradient(ellipse 80% 50% at 92% 18%, rgba(255,200,220,0.28) 0%, transparent 55%),
        radial-gradient(ellipse 70% 50% at 8% 30%, rgba(200,230,255,0.32) 0%, transparent 55%),
        linear-gradient(175deg, #f0faff 0%, #dff0f8 25%, #cce8f5 55%, #b8dff0 80%, #a8d8ee 100%) !important;
    background-attachment: fixed !important;
    color: #0a2a40 !important;
}
.gradio-container { max-width: 1200px !important; }

/* ── Frosted-glass blocks & panels ── */
.block, .form, .gr-group, .gr-box,
[class*="block"], .panel, .tabs {
    background: rgba(255,255,255,0.72) !important;
    border: 0.5px solid rgba(100,160,210,0.2) !important;
    border-radius: 14px !important;
}

/* ── Inputs & textareas ── */
input, textarea, .wrap-inner,
input[type="text"], input[type="number"] {
    background: rgba(255,255,255,0.88) !important;
    color: #0a2a40 !important;
    border-color: rgba(100,160,210,0.28) !important;
}
input::placeholder, textarea::placeholder { color: rgba(40,100,150,0.4) !important; }

/* ── All body text dark ── */
p, span, label, div, h1, h2, h3, h4, td, th, li,
.prose *, .gr-markdown * {
    color: #0a2a40 !important;
}

/* ── App header — remove double padding ── */
.app-header {
    background: rgba(255,255,255,0.8) !important;
    border-radius: 16px; padding: 18px 24px; margin: 0 !important;
    border: 0.5px solid rgba(100,160,210,0.25) !important;
    box-shadow: 0 2px 18px rgba(100,180,220,0.08), inset 0 1px 0 rgba(255,255,255,0.9);
}
.app-header h1 { font-size: 1.4rem; font-weight: 700; color: #0a2a40 !important; margin: 0; }
.app-header p  { font-size: 0.87rem; color: rgba(30,90,140,0.7) !important; margin: 4px 0 0; }
/* strip the Gradio block wrapper around the HTML component */
.gradio-container > .main > .wrap > .contain > div:first-child > .block,
.gr-prose .block { padding: 0 !important; background: transparent !important; border: none !important; box-shadow: none !important; }

/* ── Variant headers — same muted blue as labels ── */
.variant-a-label, .variant-b-label {
    color: rgba(30,80,140,0.75) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

/* ── Winner box — solid white, no semi-transparent overlay ── */
.winner-box {
    background: #ffffff !important;
    border: 0.5px solid rgba(100,160,210,0.3) !important;
    border-radius: 12px !important; padding: 18px 22px;
    opacity: 1 !important;
}
.winner-box * { opacity: 1 !important; }

/* ── Run button ── */
button.primary-btn {
    background: linear-gradient(110deg,
        rgba(220,180,100,0.22), rgba(220,130,180,0.16), rgba(120,160,230,0.2)) !important;
    border: 1px solid rgba(100,160,210,0.42) !important;
    color: #0a3050 !important; font-weight: 600 !important;
    box-shadow: 0 2px 14px rgba(100,180,220,0.14) !important;
}

/* ── Dropdown popup — light background ── */
ul[role="listbox"], ul[role="listbox"] li, ul[role="listbox"] li * {
    background: rgba(255,255,255,0.97) !important;
    color: #0a2a40 !important;
}
ul[role="listbox"] li:hover { background: rgba(200,230,250,0.5) !important; }

/* ── Remove blue pill background from all block labels ── */
/* gr.Textbox / gr.Dropdown: label > span */
.block > label > span,
.label-wrap > span,
.block label > span:first-child,
/* gr.Radio / gr.Checkbox: fieldset > legend */
.block fieldset > legend,
.block fieldset > legend > span,
.block fieldset legend span {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: rgba(100,160,210,0.18) !important;
    box-shadow: none !important;
    padding: 2px 8px !important;
    border-radius: 6px !important;
}

/* ── Segmented toggle (Text / Image) ── */
input[type="radio"] { display: none !important; }
fieldset { border: none !important; padding: 0 !important; background: transparent !important; }

/* The div Gradio wraps radio labels in — becomes the toggle track */
fieldset > div {
    display: inline-flex !important;
    gap: 0 !important;
    padding: 3px !important;
    background: rgba(200,230,250,0.25) !important;
    border: 1.5px solid rgba(46,127,184,0.35) !important;
    border-radius: 999px !important;
    box-shadow: inset 0 1px 3px rgba(100,160,210,0.12) !important;
}

/* Each label = one toggle segment */
label:has(> input[type="radio"]) {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 5px 22px !important;
    border-radius: 999px !important;
    border: none !important;
    cursor: pointer !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: rgba(30,80,140,0.7) !important;
    background: transparent !important;
    box-shadow: none !important;
    transition: background 0.18s ease, color 0.18s ease !important;
    white-space: nowrap !important;
}

/* Active segment */
label:has(> input[type="radio"]:checked) {
    background: #ffffff !important;
    color: #0a2a40 !important;
    box-shadow: 0 1px 4px rgba(100,160,210,0.25) !important;
}

/* ── Space Mono for block labels (Textbox/Dropdown + Radio/Checkbox) ── */
.block label > span:first-child,
.block fieldset > legend,
.block fieldset > legend > span {
    font-family: 'Space Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(30,80,140,0.6) !important;
}

/* ── Image / file upload area — white background ── */
.upload-container, [data-testid="image"], .svelte-upload,
.gr-file-preview, .image-frame,
.upload-container > div,
input[type="file"] + div,
.wrap.svelte-z7cif2, .wrap.svelte-gwtpzf,
.wrap > .image-container,
[class*="upload"] > [class*="wrap"],
.block .wrap { background: #ffffff !important; }

/* ── Inline code ── */
code, pre, code * {
    background: rgba(200,230,250,0.35) !important;
    color: #0a2a40 !important;
    border-radius: 4px !important;
    padding: 1px 5px;
}

/* ── Results / markdown — force dark text everywhere ── */
.gr-markdown, .gr-markdown *,
.prose, .prose *,
.md, .md *,
[data-testid="markdown"], [data-testid="markdown"] *,
.output-markdown, .output-markdown *,
.block .prose p, .block .prose span,
.block .prose strong, .block .prose em,
.block .prose h1, .block .prose h2, .block .prose h3, .block .prose h4,
.block .prose li, .block .prose blockquote {
    color: #0a2a40 !important;
}

/* ── Tables in results — dark text + light row bg ── */
table { border-collapse: collapse !important; width: 100% !important; }
table th {
    background: rgba(200,230,250,0.4) !important;
    color: #0a2a40 !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em !important;
    border-bottom: 1px solid rgba(100,160,210,0.3) !important;
}
table td {
    color: #0a2a40 !important;
    padding: 7px 12px !important;
    border-bottom: 0.5px solid rgba(100,160,210,0.15) !important;
    font-size: 0.85rem !important;
    background: rgba(255,255,255,0.45) !important;
}
table tr:hover td { background: rgba(200,230,250,0.25) !important; }

/* ── Hide progress bar entirely ── */
.progress-bar-wrap,
.progress-level,
div[class*="progress-bar"],
div[class*="progress_bar"],
.eta-bar,
.meta-text,
.meta-text-center { display: none !important; }

/* ── Hide Gradio footer ── */
footer { display: none !important; }

/* ── All layout containers — white backgrounds ──
   Uses compound selector html+body+.gradio-container (specificity 0,1,2)
   + div element (0,0,1) = total (0,1,3) which beats Gradio's two-Svelte-class (0,2,0) ── */
html body .gradio-container div,
html body .gradio-container section,
html body .gradio-container .form,
html body .gradio-container .contain,
html body .gradio-container .gap {
    background: rgba(255,255,255,0.85) !important;
}
/* Preserve transparent backgrounds on purely structural/wrapper elements */
html body .gradio-container .tabs,
html body .gradio-container nav { background: transparent !important; }

/* Legacy single-class fallback */
.block, .form, .gr-group, .gr-box,
[class*="block"], .panel,
[data-testid="row"], [data-testid="column"], [data-testid="group"] {
    background: rgba(255,255,255,0.85) !important;
}

/* ── Hide Share and Fullscreen buttons on plot components ── */
button[title="Share"], button[aria-label="Share"],
button[title="share"], a[title="Share"],
[data-testid="share-button"] { display: none !important; }

button[title="Full Screen"], button[title="Fullscreen"],
button[aria-label="Full Screen"], button[aria-label="Fullscreen"],
button[aria-label="full screen"], a[title="Full Screen"],
[data-testid="fullscreen-button"] { display: none !important; }

/* ── All icon/action buttons — white background ── */
.icon-button, .icon-buttons button,
button.icon-button,
[data-testid="download-button"],
button[title="Download"], button[aria-label="Download"],
button[title="download"], a[title="Download"] {
    background: #ffffff !important;
    border: 0.5px solid rgba(100,160,210,0.3) !important;
    border-radius: 8px !important;
    color: #0a2a40 !important;
    box-shadow: 0 1px 6px rgba(100,160,210,0.1) !important;
}
.icon-button svg, .icon-buttons button svg,
button.icon-button svg,
button[title="Download"] svg, button[aria-label="Download"] svg,
[data-testid="download-button"] svg {
    stroke: #0a2a40 !important;
    fill: none !important;
    color: #0a2a40 !important;
}

/* ── Results panel via inner div ID (#tr-results is inside the HTML content) ── */
#tr-results,
#tr-results * { color: #0a2a40 !important; opacity: 1 !important; background: transparent; }
#tr-results { background: #ffffff !important; font-size: .88rem !important; line-height: 1.6 !important; }
#tr-results h2 { font-size: 1.1rem !important; font-weight: 700 !important; color: #0a2a40 !important; }
#tr-results h3 { font-size: .7rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: .1em !important; color: rgba(30,80,140,.85) !important; }
#tr-results p { color: #0a2a40 !important; }
#tr-results th { background: rgba(200,230,250,.5) !important; color: #0a2a40 !important; font-weight: 600 !important; padding: 6px 10px !important; border-bottom: 1px solid rgba(100,160,210,.3) !important; font-size: .73rem !important; text-align: left !important; }
#tr-results td { background: #ffffff !important; color: #0a2a40 !important; padding: 6px 10px !important; border-bottom: .5px solid rgba(100,160,210,.18) !important; }
#tr-results .conf-hi { color: #16a34a !important; font-weight: 600 !important; }
#tr-results .conf-med { color: #b45309 !important; font-weight: 600 !important; }
#tr-results .conf-lo { color: #dc2626 !important; font-weight: 600 !important; }
#tr-results .tr-a { color: #1d4ed8 !important; font-weight: 600 !important; }
#tr-results .tr-b { color: #c2410c !important; font-weight: 600 !important; }
#tr-results .warn { color: #0a2a40 !important; background: rgba(255,237,213,.4) !important; }

/* ── Winner output — elem_id="winner-output" on the gr.HTML component wrapper ──
   This ID is set by Gradio on the outermost wrapper div (never sanitized).
   Specificity (1,0,0) beats every Svelte-scoped class Gradio uses.           ── */
#winner-output,
#winner-output > *,
#winner-output .wrap,
#winner-output .html-container {
    background: #ffffff !important;
    color: #0a2a40 !important;
}
#winner-output * { color: #0a2a40 !important; }

/* heading styles inside results */
#winner-output h2 {
    font-size: 1.1rem !important; font-weight: 700 !important;
    margin: 0 0 3px !important; color: #0a2a40 !important;
}
#winner-output h3 {
    font-size: 0.7rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    margin: 16px 0 6px !important; color: rgba(30,80,140,0.85) !important;
}
#winner-output p { font-size: 0.78rem !important; margin: 3px 0 !important; color: #0a2a40 !important; }

/* tables inside results */
#winner-output table {
    border-collapse: collapse !important; width: 100% !important;
    margin: 4px 0 10px !important; font-size: 0.84rem !important;
}
#winner-output th {
    background: rgba(200,230,250,0.5) !important; color: #0a2a40 !important;
    font-weight: 600 !important; padding: 6px 10px !important;
    border-bottom: 1px solid rgba(100,160,210,0.3) !important;
    font-size: 0.73rem !important; letter-spacing: 0.04em !important; text-align: left !important;
}
#winner-output td {
    background: #ffffff !important; color: #0a2a40 !important;
    padding: 6px 10px !important; border-bottom: 0.5px solid rgba(100,160,210,0.18) !important;
    text-align: left !important;
}
#winner-output td.ra { text-align: right !important; }
#winner-output .cv {
    background: rgba(200,230,250,0.5) !important; color: #0a2a40 !important;
    border-radius: 4px !important; padding: 1px 6px !important; font-size: 0.78em !important;
}
#winner-output .conf-hi { color: #16a34a !important; font-weight: 600 !important; }
#winner-output .conf-med { color: #b45309 !important; font-weight: 600 !important; }
#winner-output .conf-lo { color: #dc2626 !important; font-weight: 600 !important; }
#winner-output .tr-a { color: #1d4ed8 !important; font-weight: 600 !important; }
#winner-output .tr-b { color: #c2410c !important; font-weight: 600 !important; }
#winner-output .tr-tie { color: #4b5563 !important; }
#winner-output .warn {
    border-left: 3px solid rgba(180,83,9,0.45) !important;
    padding: 4px 10px !important; margin: 4px 0 !important;
    background: rgba(255,237,213,0.4) !important;
    border-radius: 0 6px 6px 0 !important; font-size: 0.8rem !important; color: #0a2a40 !important;
}
"""


def _variant_column(title_class: str, title_text: str, default_name: str):
    """Returns name, mode, and input component references for one variant."""
    gr.HTML(f"<h3 class='{title_class}'>{title_text}</h3>")
    name = gr.Textbox(label="Name (optional)", value=default_name)
    mode = gr.Radio(
        label="Input modality",
        choices=ALL_MODALITY_CHOICES,
        value=ModalityMode.TEXT.value,
    )
    with gr.Group(visible=True) as grp_text:
        tb = gr.Textbox(
            label="Text stimulus",
            placeholder="Long-form OK — ad copy, transcript, etc.",
            lines=6,
        )
    with gr.Group(visible=False) as grp_image:
        image = gr.Image(
            label="Image (.jpg, .png, .webp) — converted to an 8-s static clip",
            type="filepath",
            sources=["upload"],
        )
    with gr.Group(visible=False) as grp_audio:
        audio = gr.Audio(
            label="Audio (.wav, .mp3, .flac, .ogg)",
            type="filepath",
            sources=["upload"],
        )
    with gr.Group(visible=False) as grp_video:
        video = gr.Video(
            label="Video (.mp4, .avi, .mkv, .mov, .webm)",
            sources=["upload"],
        )

    mode.change(
        _toggle_input_groups,
        inputs=[mode],
        outputs=[grp_text, grp_image, grp_audio, grp_video],
    )
    return name, mode, tb, image, audio, video


GUIDE_MARKDOWN = """
# How to use the TRIBE A/B Brain Simulator

## What this is
This tool predicts how a typical viewer\'s brain would respond to two creative
variants. Under the hood it uses **TRIBE v2** (Meta FAIR, d\'Ascoli 2025) - a
deep neural network that predicts fMRI brain activity from naturalistic video,
audio, and text stimuli. TRIBE won the 2025 Algonauts brain-encoding competition.

For each variant we simulate ~20,000 cortical surface points across a typical
viewer\'s brain at 1-second resolution, then aggregate that into the gauges,
region bars, and brain-surface images you see below.

## How to run a test
1. For each variant, pick an input modality (Text / Image / Audio / Video).
2. Provide the stimulus - paste copy, upload an image, an audio file, or a video clip.
3. Hit **Run A/B Neural Test** and wait.

### Expected latency per test (both variants combined)

| Modality | Wall time |
|---|---|
| Text | ~40-80 s |
| Image | ~2 min (image becomes an 8-s static clip internally) |
| Audio | ~6-12 min (whisperx ASR is the bottleneck) |
| Video | ~12-30 min (V-JEPA2 vision encoder is the bottleneck) |

Avoid mixing modalities across A and B - comparing a text variant against
a video variant is technically possible but the model\'s response patterns
differ enough that the numbers stop being directly comparable.

## How to read the results

**Winner box** (top): the headline call. Names the winner on the combined
proxy score, with confidence (High / Medium / Low based on statistical effect
size). Read this first; everything else is supporting evidence.

**Activation over time** (line chart): mean brain activation per second of
stimulus exposure. Look for the *shape* of the curve, not just the average:

- Ramp \u2192 sustain \u2192 high finish = sticky
- Spike \u2192 flatline = "hook then collapse"
- Flat = the viewer never engaged

**Region comparison** (12 bars, A vs B): average activation per coarse brain
system. The ones that matter most for production:

| Region | What it tracks | Bigger = |
|---|---|---|
| Visual / Occipital | Perception, image detail | Visual is doing work |
| Auditory / Temporal | Speech, music, sound design | Audio is landing |
| Language Areas | Semantic word comprehension | Copy is being processed |
| Prefrontal Cortex | Attention, \"is this for me?\" | The viewer cares |
| Limbic / Emotional | Emotional load | More likely to be shared |
| **Default Mode Network** | Mind-wandering, disengagement | **High DMN is bad** - viewer checked out |
| Attention Network | Focused selective attention | The viewer is locked in |

**ROI systems** (3 bars): the 12 regions collapsed into Visual / Auditory /
Language. Use this for a quick "did the visual, the sound, or the words do
the work?" diagnostic.

**Proxy scores (radar + gauges)**:

- **Engagement** - overall cortical activation, weighted toward executive and attention regions.
- **Attention** - focused-attention regions; penalised by DMN.
- **Emotion** - limbic system weighted heavily; predicts emotional resonance and sharing likelihood.

These are derived from region activations, not direct TRIBE outputs.

**Brain surface images** (bottom): max-over-time activation on the lateral
view of a standard cortical surface, one per variant. The hot spots show
where the stimulus drove the strongest peak response. Useful when discussing
with a creative director who wants to see *where* in the brain the variant
lit up.

## Important caveats - read these before pitching results to a client

- **This is a model prediction, not a measurement.** TRIBE was trained on
  publicly available fMRI from ~4 subjects watching movies and clips.
  Validation Pearson r is ~0.22 for text-only inputs and ~0.31 for full
  multimodal (paper section 3.3). Treat numbers as directional, not diagnostic.
- **No demographic targeting.** Predictions are for a "typical" subject;
  no controls for age, gender, culture, or audience segment.
- **Short stimuli are noisier.** Single words or single still images
  produce shaky time-series. Use at least a sentence, an 8-s clip, or a
  paragraph.
- **Not a substitute for actual user testing.** Use it as a fast pre-filter
  to pick which 2-3 variants to put in front of real eyes - not as a
  replacement for them.

## What modality should I pick?

| Pick | Best for |
|---|---|
| **Text** | Ad copy, headlines, scripts, captions, hooks, narrative beats |
| **Image** | Hero shots, thumbnails, key visuals, poster comps |
| **Audio** | VO reads, jingles, music beds, podcast intros |
| **Video** | 15s-2min finished cuts, animatics, storyboards as motion |

## What to do with a result

- **Big gap, high confidence** \u2192 ship the winner.
- **Big gap, low confidence** \u2192 run again with related variations to separate
  "the idea won" from "the wording won".
- **Small gap, any confidence** \u2192 it\'s a coin flip. Decide on production cost
  or other criteria. Don\'t relitigate.
- **Variant with high DMN** \u2192 the brain checks out. Re-cut.
- **Variant with high Limbic but low Attention** \u2192 emotionally loud but
  forgettable. Tighten focus.
"""

def build_ui() -> gr.Blocks:
    _theme = gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="sky",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        block_label_background_fill="white",
        block_label_background_fill_dark="white",
        block_label_border_color="rgba(100,160,210,0.2)",
        block_label_border_color_dark="rgba(100,160,210,0.2)",
        block_label_text_color="rgba(30,80,140,0.6)",
        block_label_text_color_dark="rgba(30,80,140,0.6)",
        input_background_fill="white",
        input_background_fill_dark="white",
    )

    with gr.Blocks(
        theme=_theme,
        css=CUSTOM_CSS,
        title="A/B Test Simulator — TRIBE v2",
    ) as demo:
        gr.HTML(
            """
        <div class="app-header">
            <h1>🧠 TRIBE A/B Brain Simulator</h1>
            <p>Predict how a typical viewer's brain responds to two creative variants — text, image, audio, or video.</p>
        </div>
        """
        )

        with gr.Tabs():
            with gr.Tab("A/B Test"):
                with gr.Row():
                    preset_dd = gr.Dropdown(
                        choices=PRESET_NAMES,
                        value="(none)",
                        label="Example preset (fills modes + text)",
                        scale=2,
                    )
                preset_hint = gr.Markdown("")
        
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        name_a, mode_a, text_a, image_a, audio_a, video_a = _variant_column(
                            "variant-a-label", "Variant A", "Variant A"
                        )
                    with gr.Column(scale=1):
                        name_b, mode_b, text_b, image_b, audio_b, video_b = _variant_column(
                            "variant-b-label", "Variant B", "Variant B"
                        )
        
                run_btn = gr.Button("🚀  Run A/B Neural Test", variant="primary", elem_classes=["primary-btn"])
        
                gr.Markdown("### Results")
                winner_md = gr.HTML(visible=False, elem_classes=["winner-box"], elem_id="winner-output")
                with gr.Row():
                    prev_a_md = gr.Markdown(label="Preview A")
                    prev_b_md = gr.Markdown(label="Preview B")
        
                with gr.Group(visible=False) as results_group:
                    gr.Markdown("### Activation (time-aligned)")
                    timeline_plot = gr.Plot(show_label=False)
                    region_plot = gr.Plot(show_label=False)
                    roi_plot = gr.Plot(show_label=False)
                    radar_plot = gr.Plot(show_label=False)
                    with gr.Row():
                        gauge_eng = gr.Plot(show_label=False)
                        gauge_att = gr.Plot(show_label=False)
                        gauge_emo = gr.Plot(show_label=False)
                    gr.Markdown("### Brain surfaces (max over time)")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Variant A**")
                            brain_a_img = gr.Image(label="", type="pil")
                        with gr.Column():
                            gr.Markdown("**Variant B**")
                            brain_b_img = gr.Image(label="", type="pil")
                    export_csv = gr.File(label="CSV export")
        
            with gr.Tab("Guide"):
                gr.Markdown(GUIDE_MARKDOWN)

        # Wiring
        preset_dd.change(
            load_preset,
            inputs=[preset_dd],
            outputs=[mode_a, mode_b, text_a, text_b, name_a, name_b, preset_hint],
        )

        # Two-way modality sync
        mode_a.change(fn=lambda v: gr.update(value=v), inputs=[mode_a], outputs=[mode_b])
        mode_b.change(fn=lambda v: gr.update(value=v), inputs=[mode_b], outputs=[mode_a])

        ab_outputs = [
            winner_md,
            prev_a_md,
            prev_b_md,
            timeline_plot,
            region_plot,
            roi_plot,
            radar_plot,
            gauge_eng,
            gauge_att,
            gauge_emo,
            brain_a_img,
            brain_b_img,
            export_csv,
            results_group,
        ]
        run_btn.click(
            fn=run_ab_test,
            inputs=[
                mode_a, name_a, text_a, image_a, audio_a, video_a,
                mode_b, name_b, text_b, image_b, audio_b, video_b,
            ],
            outputs=ab_outputs,
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRIBE v2 A/B Test Simulator")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--cache-dir", default="./cache")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine = TribeInferenceEngine(cache_dir=args.cache_dir, device="auto")
    build_ui().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
