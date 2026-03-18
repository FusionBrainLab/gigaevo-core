"""Validate dashboard HTML using Playwright screenshot + Gemini 3.1 Pro vision critic.

Scoring pipeline:
1. Write HTML to /tmp
2. Render with Playwright Chromium headless (1440x900, wait 2s for JS)
3. Take full-page screenshot
4. Send screenshot to Gemini 3.1 Pro (via OpenRouter) with scoring rubric
5. Return averaged score as fitness in [0, 1]

HTML artifacts are saved to DASHBOARD_HTML_DIR (default: /tmp/gigaevo_dashboard_outputs/)
so you can open them in a browser during/after the run.
Each file is named: fitness{score:.4f}_{hash}.html
"""

import base64
import hashlib
import os
from pathlib import Path
import tempfile
import time

from pydantic import BaseModel, Field

# Directory where scored HTML files are saved for browsing.
# Override with env var DASHBOARD_HTML_DIR.
_DEFAULT_HTML_DIR = "/tmp/gigaevo_dashboard_outputs"


# ---------------------------------------------------------------------------
# Hard validation (fast, before any rendering)
# ---------------------------------------------------------------------------


def _hard_validate(html: str) -> None:
    """Raise ValueError if HTML is structurally broken."""
    if not isinstance(html, str):
        raise ValueError(f"entrypoint must return str, got {type(html).__name__}")
    if len(html) < 500:
        raise ValueError(f"HTML too short ({len(html)} chars); minimum 500")
    if "<html" not in html.lower():
        raise ValueError("HTML missing <html> tag")


# ---------------------------------------------------------------------------
# Playwright rendering
# ---------------------------------------------------------------------------


def _render_screenshot(html: str) -> bytes:
    """Render HTML in Chromium and return full-page PNG bytes."""
    from playwright.sync_api import sync_playwright

    # Write to temp file (file:// URL avoids network issues)
    h = hashlib.md5(html.encode()).hexdigest()[:12]
    html_path = Path(tempfile.gettempdir()) / f"gigaevo_dashboard_{h}.html"
    png_path = Path(tempfile.gettempdir()) / f"gigaevo_dashboard_{h}.png"

    try:
        html_path.write_text(html, encoding="utf-8")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            # Allow external CDN requests (Chart.js, D3, Plotly, etc.).
            # Use a 15s timeout — if a CDN is unreachable the program is invalid.
            try:
                page.goto(
                    f"file://{html_path}",
                    wait_until="domcontentloaded",
                    timeout=15000,
                )
            except Exception as e:
                browser.close()
                raise ValueError(f"Page load timed out (CDN unreachable?): {e}") from e
            # Collapse all CSS animation/transition durations so entrance animations
            # (e.g. opacity:0 → 1 fade-ins) complete instantly before screenshot.
            page.add_style_tag(
                content=(
                    "*, *::before, *::after {"
                    "  animation-duration: 0.001s !important;"
                    "  animation-delay: 0s !important;"
                    "  transition-duration: 0.001s !important;"
                    "  transition-delay: 0s !important;"
                    "}"
                )
            )
            # Wait for JS/canvas rendering to settle
            page.wait_for_timeout(2000)
            screenshot_bytes = page.screenshot(full_page=True)
            browser.close()

        png_path.write_bytes(screenshot_bytes)
        return screenshot_bytes

    finally:
        # Cleanup temp files
        for p in [html_path, png_path]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def _is_blank_screenshot(png_bytes: bytes) -> bool:
    """Return True if screenshot is nearly uniform (all one colour = blank/crash)."""
    try:
        import zlib

        # Parse PNG IDAT to sample pixels — approximate check via gzip entropy
        # If image is almost entirely one colour, compressed size << raw size
        raw_size = len(png_bytes)
        compressed = zlib.compress(png_bytes)
        # If compression ratio > 0.99 (near-perfect), likely blank
        return len(compressed) / raw_size > 0.97
    except Exception:
        return False


# ---------------------------------------------------------------------------
# VLM scoring
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a BRUTALLY strict UI/UX critic scoring dashboards for an AI research lab.
Your default score for any dimension is 3. Increase ONLY if you see clear evidence of excellence.
Mediocre work scores 3–4. Good work scores 5–6. Impressive work scores 7–8. Breathtaking scores 9–10.
You must NEVER give a score above 5 unless the feature is clearly, visually present and well-executed.
When in doubt, score LOWER. Inflated scores defeat the purpose of the evaluation.
"""

_SCORING_PROMPT = """\
Score this GigaEvo monitoring dashboard screenshot on exactly 5 dimensions (0–10 each).

ANCHOR — a plain baseline dashboard (colored divs, basic canvas line chart, flat archive strip, unstyled table, no animations) MUST score: visual_appeal=3, multi_run_clarity=4, data_completeness=4, interactivity_cues=2, innovation=2. If what you see looks like that baseline, give those scores. Only score higher if you see clear improvements.

Score on exactly these 5 dimensions (0–10 each). Use the FULL range — do NOT cluster at 7–8.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. visual_appeal  —  Does this look like it was designed by a world-class UI studio?
   0–2: Ugly. Raw HTML table aesthetic, mismatched colors, no visual polish.
   3–4: Basic. Dark background ticked, but flat, boring, like a default Bootstrap theme.
   5–6: Acceptable. Consistent color usage, readable, but nothing special.
   7:   Good. Clear AIRI brand colors (#2FBEAD teal, #E5434D coral, #161C27 bg), clean layout, pleasing.
   8:   Impressive. Depth through layering, subtle gradients, beautiful typography, polished spacing.
   9:   Stunning. Feels like a premium product — glows, layered shadows, micro-detail, pixel-perfect.
   10:  Museum-worthy. Every pixel intentional. Rivals the best commercial monitoring tools (Grafana+).

2. multi_run_clarity  —  Can you understand ALL 4 experiments instantly, at a glance?
   0–2: Can't tell which panel belongs to which run. No labels or labels unreadable.
   3–4: Panels exist but visual distinction is weak — same colors, hard to compare.
   5–6: Runs are labeled and separated but comparison requires effort.
   7:   Clear per-run identity (distinct label/badge), status visible, easy to scan all 4 runs.
   8:   Excellent hierarchy — stalled runs visually alarming (coral), complete runs muted, running runs prominent.
   9:   Instant at-a-glance comprehension. Side-by-side comparison is effortless. Best-run highlighted.
   10:  Perfect information architecture. A non-expert could understand the full state in 3 seconds.

3. data_completeness  —  Are ALL required data elements present and readable for EVERY run?
   THE MOST IMPORTANT ELEMENT is the fitness trajectory chart with TWO clearly labeled lines:
   - gen_fitness_frontier: bold bright teal line (running best — must be monotone non-decreasing)
   - gen_fitness_mean: thinner line (population average per generation)
   If BOTH lines are not clearly visible and labeled on the fitness chart, this dimension CANNOT exceed 4.

   Required per run:
   (a) fitness chart with BOTH frontier (best-ever) AND mean lines, clearly labeled — CRITICAL
   (b) 1D archive strip — 150 horizontal bins colored by fitness (empty=dark, high fitness=bright teal)
   (c) generation counter + progress bar
   (d) status badge (running / stalled / complete)
   (e) best fitness value
   (f) valid rate AND invalid program count (failed validations must be visible, not hidden)
   (g) top-5 programs table (rank, fitness, generation, mutation, code preview)
   (h) genealogy showing ancestor chain for the best program
   0–2: No fitness chart or only one line. Most elements missing.
   3–4: Fitness chart exists but shows only ONE line (either frontier OR mean, not both).
   5–6: Both lines present but not labeled. Archive strip absent or not color-coded.
   7:   Both frontier+mean clearly labeled. All 8 elements present. Archive teal gradient visible.
   8:   All present + frontier vs mean legend, archive bin count, invalid count prominent.
   9:   All present, richly detailed. Archive color legend. Top programs show mutation names. Genealogy.
   10:  Complete, beautiful, immediately useful. Both chart lines unmissable.

4. interactivity_cues  —  Does the design FEEL alive and interactive?
   0–2: Completely static. Plain HTML with no visual motion or affordance.
   3–4: One CSS transition or a hover color change. Feels like an afterthought.
   5–6: Some animations present (e.g. progress bar fill, chart draw-on). Hover states on a few elements.
   7:   Noticeable animated elements — pulsing stalled runs, entrance animations, chart lines drawing in.
   8:   Rich interactive feel — archive bin tooltips on hover, animated fitness curves, hover highlights on panels.
   9:   Highly polished motion design. Animations are purposeful, not distracting. Feels like a live app.
   10:  Fully alive. Real-time feel with smooth transitions everywhere. Indistinguishable from a live product.

5. innovation  —  Does this push beyond a generic dashboard template?
   0–2: Literally a table or basic divs. No creative choices.
   3–4: Standard line chart + status text. Looks like every other dashboard.
   5–6: Some personality — maybe a radial progress ring or gradient fills, but nothing surprising.
   7:   One genuinely clever design idea (e.g. arc gauge for valid_rate, sparklines in headers, animated archive).
   8:   Multiple creative choices that serve the data. The design makes you think "I haven't seen this before."
   9:   Inventive layout or visualization that feels purpose-built for evolutionary algorithm monitoring.
   10:  Genuinely novel. A design that could be featured in a UI showcase. Memorable.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CALIBRATION EXAMPLES:
- Plain baseline (colored divs, canvas chart, unstyled table, no animations, flat archive strip): 3, 4, 4, 2, 2 → fitness ≈ 0.30
- Decent dark theme with styled cards and labeled charts but no archive strip or genealogy: 5, 5, 4, 3, 3 → fitness ≈ 0.40
- Good Grafana-style (dark theme, all data elements present, some hover states): 6, 7, 6, 5, 4 → fitness ≈ 0.56
- Impressive (beautiful typography, archive strip with gradient, animated charts, all 8 data elements): 8, 7, 8, 7, 7 → fitness ≈ 0.74
- Exceptional (stunning visuals, perfect information architecture, all elements, polished animations): 9, 9, 9, 8, 8 → fitness ≈ 0.86

If the dashboard does NOT have a 1D archive strip with fitness-colored bins, data_completeness CANNOT exceed 4.
If there are NO hover effects or animations visible, interactivity_cues CANNOT exceed 3.
If the layout looks like a standard Bootstrap/generic template, innovation CANNOT exceed 3.
"""

_SCORE_KEYS = [
    "visual_appeal",
    "multi_run_clarity",
    "data_completeness",
    "interactivity_cues",
    "innovation",
]


class DashboardScores(BaseModel):
    """Structured VLM critic output — one integer score per dimension (0–10)."""

    visual_appeal: int = Field(ge=0, le=10)
    multi_run_clarity: int = Field(ge=0, le=10)
    data_completeness: int = Field(ge=0, le=10)
    interactivity_cues: int = Field(ge=0, le=10)
    innovation: int = Field(ge=0, le=10)


def _resolve_api_key() -> str:
    """Return OPENAI_API_KEY, falling back to .env if the env var is absent/placeholder.

    load_dotenv() in run.py won't override a pre-existing shell var like
    OPENAI_API_KEY=EMPTY, so exec_runner subprocesses need this fallback.
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    if key and key.upper() != "EMPTY":
        return key
    from dotenv import dotenv_values

    for candidate in [
        os.path.join(os.getcwd(), ".env"),  # repo root (exec_runner cwd = project root)
        ".env",
    ]:
        vals = dotenv_values(os.path.abspath(candidate))
        if vals.get("OPENAI_API_KEY"):
            return vals["OPENAI_API_KEY"]
    raise RuntimeError(
        "OPENAI_API_KEY not set. Add it to .env or export it before launching."
    )


def _score_with_vlm(png_bytes: bytes) -> dict[str, float]:
    """Send screenshot to Gemini 3.1 Pro via OpenRouter and return raw scores dict.

    Uses LangChain with_structured_output(DashboardScores) + Langfuse LangChain
    callback — the same pattern as all other GigaEvo LLM calls. Stable and traced.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langfuse.langchain import CallbackHandler

    api_key = _resolve_api_key()

    llm = ChatOpenAI(
        model="google/gemini-3.1-pro-preview",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=8192,
    )
    structured_llm = llm.with_structured_output(DashboardScores)

    b64 = base64.b64encode(png_bytes).decode("ascii")
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": _SCORING_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ]
        ),
    ]

    config = {}
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        handler = CallbackHandler()
        config = {"callbacks": [handler]}

    # Retry on transient OpenRouter failures (0-token responses, timeouts).
    for attempt in range(3):
        try:
            scores: DashboardScores = structured_llm.invoke(messages, config=config)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)

    return {k: float(getattr(scores, k)) for k in _SCORE_KEYS}


# ---------------------------------------------------------------------------
# Main validate entry point
# ---------------------------------------------------------------------------


def validate(context: dict, html: str) -> dict:
    """Validate an evolved dashboard HTML.

    Args:
        context: Mock run data dict (from context.py build_context).
        html: Complete HTML string returned by entrypoint(context).

    Returns:
        {"fitness": float in [0,1], "is_valid": 1.0}

    Raises:
        ValueError: On structural HTML failures or blank screenshot.
    """
    # 1. Hard validation (cheap)
    _hard_validate(html)

    # 2. Render with Playwright
    png_bytes = _render_screenshot(html)

    # 3. Blank-screen check
    if _is_blank_screenshot(png_bytes):
        raise ValueError(
            "Screenshot appears blank (all one colour) — likely a render crash"
        )

    # 4. VLM scoring
    scores = _score_with_vlm(png_bytes)

    # 5. Average → normalize to [0, 1]  (each dim is 0–10)
    avg = sum(scores.values()) / (10.0 * len(scores))
    fitness = round(max(0.0, min(1.0, avg)), 4)

    # 6. Save HTML artifact for browsing
    _save_html_artifact(html, fitness)

    return {"fitness": fitness, "is_valid": 1.0}


def _save_html_artifact(html: str, fitness: float) -> None:
    """Save HTML to DASHBOARD_HTML_DIR for offline browsing."""
    out_dir = Path(os.environ.get("DASHBOARD_HTML_DIR", _DEFAULT_HTML_DIR))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(html.encode()).hexdigest()[:8]
        filename = f"fitness{fitness:.4f}_{h}.html"
        (out_dir / filename).write_text(html, encoding="utf-8")
    except Exception:
        pass  # Never let artifact saving crash a validation
