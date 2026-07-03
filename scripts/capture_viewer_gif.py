"""Capture a cinematic GIF from the running AQUINAS Bridge Viewer.

The camera follows a fixed sequence:
  1. Establish  — wide overhead shot drifts in
  2. Orbit      — smooth sweep around the front face
  3. Approach   — descends close to sensor level
  4. Pan        — tracking shot reveals both spans
  5. Reveal     — pulls back dramatically from far side
  6. Return     — settles back to opening angle

Prerequisites
-------------
- `aquinas viz open` serving on http://127.0.0.1:8765
- playwright + chromium: pip install playwright && playwright install chromium
- Pillow: pip install Pillow   (already in venv)

Usage
-----
python scripts/capture_viewer_gif.py [--url URL] [--out PATH]
"""
from __future__ import annotations

import argparse
import asyncio
import io
import math
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_URL = "http://127.0.0.1:8765/index.html"
DEFAULT_OUT = "docs/figures/viewer.gif"
TOTAL_FRAMES = 48          # fixed cinematic sequence
FPS          = 12          # → 4 s loop
# Viewport must be > 1280px — the viewer's responsive CSS collapses the
# WebGL canvas at ≤1280px, producing blank frames in headless capture.
VIEWPORT_W   = 1920
VIEWPORT_H   = 1080
# Output at half the viewport size: quality WebGL render, reasonable file.
GIF_W        = 1280
GIF_H        = 720
METRIC       = "health_anomaly"


# ── camera keyframe helpers ────────────────────────────────────────────────────

def _ss(t: float) -> float:
    """Smoothstep easing."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _slerp(a: float, b: float, t: float) -> float:
    return _lerp(a, b, _ss(t))


def _sphere(angle_deg: float, radius: float, height: float,
            tx: float = 45.0, ty: float = 0.0, tz: float = 0.0):
    """Return ((cx,cy,cz), (tx,ty,tz)) for a spherical camera position."""
    a = math.radians(angle_deg)
    cx = tx + radius * math.cos(a)
    cy = ty + height
    cz = tz + radius * math.sin(a)
    return (cx, cy, cz), (tx, ty, tz)


def build_keyframes() -> list[tuple[tuple, tuple]]:
    """Return TOTAL_FRAMES (cam_pos, target) pairs for the cinematic sequence."""
    kf: list[tuple[tuple, tuple]] = []

    # ── Phase 0: Establish (8 fr) — drift in from high-wide angle ──────────
    for i in range(8):
        t = i / 7
        cam, tgt = _sphere(_slerp(20, 35, t), _slerp(82, 58, t), _slerp(42, 24, t))
        kf.append((cam, tgt))

    # ── Phase 1: Orbit front face (10 fr) — 35° → 125° ─────────────────────
    for i in range(10):
        t = i / 9
        cam, tgt = _sphere(_slerp(35, 125, t), _slerp(58, 52, t), _slerp(24, 16, t))
        kf.append((cam, tgt))

    # ── Phase 2: Approach (8 fr) — 125°→145°, descend to sensor level ──────
    for i in range(8):
        t = i / 7
        cam, tgt = _sphere(_slerp(125, 145, t), _slerp(52, 28, t), _slerp(16, 7, t))
        kf.append((cam, tgt))

    # ── Phase 3: Pan (8 fr) — slide along bridge at close range ─────────────
    a_rad = math.radians(145)
    r = 28
    for i in range(8):
        t = i / 7
        tx = _slerp(12.0, 78.0, t)
        cx = tx + r * math.cos(a_rad)
        cz =      r * math.sin(a_rad)
        kf.append(((cx, 7.0, cz), (tx, 0.0, 0.0)))

    # ── Phase 4: Reveal (10 fr) — pull back orbiting far side 145°→275° ─────
    for i in range(10):
        t = i / 9
        cam, tgt = _sphere(_slerp(145, 275, t), _slerp(28, 65, t), _slerp(7, 30, t))
        kf.append((cam, tgt))

    # ── Phase 5: Return (4 fr) — settle near opening angle ──────────────────
    for i in range(4):
        t = i / 3
        cam, tgt = _sphere(_slerp(275, 380, t), _slerp(65, 58, t), _slerp(30, 24, t))
        kf.append((cam, tgt))

    assert len(kf) == TOTAL_FRAMES, f"Expected {TOTAL_FRAMES} frames, got {len(kf)}"
    return kf


# ── capture ────────────────────────────────────────────────────────────────────

async def capture(url: str, out: Path) -> None:
    keyframes = build_keyframes()
    rgb_frames: list[Image.Image] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=[
                "--enable-webgl",
                "--use-gl=angle",
                "--use-angle=swiftshader-webgl",
                "--ignore-gpu-blocklist",
                "--disable-gpu-sandbox",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security",
            ]
        )
        page = await browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})

        print(f"Loading {url} …")
        await page.goto(url, wait_until="networkidle", timeout=90_000)

        # Wait for the 3-D scene + API to be ready.
        await page.wait_for_function(
            "typeof window.__viewer !== 'undefined' && window.__viewer.camera !== null",
            timeout=40_000,
        )

        # ── Kill all CSS transitions so UI elements don't flicker between frames.
        await page.add_style_tag(content=(
            "*, *::before, *::after {"
            "  transition-duration: 0s !important;"
            "  transition-delay: 0s !important;"
            "  animation-duration: 0s !important;"
            "  animation-delay: 0s !important;"
            "}"
        ))

        # Switch to health metric.
        try:
            await page.evaluate(f"window.__viewer.setMetric('{METRIC}')")
        except Exception:
            pass
        await page.wait_for_timeout(2_000)

        # Inject a tight camera-setter so each frame only needs one evaluate call.
        await page.evaluate("""
            window.__setCam = function(cx, cy, cz, tx, ty, tz) {
                const cam  = window.__viewer.camera;
                const ctrl = window.__viewer.controls;
                cam.position.set(cx, cy, cz);
                ctrl.target.set(tx, ty, tz);
                ctrl.update();
                if (window.__viewer.render) window.__viewer.render();
            };
        """)

        print(f"Capturing {TOTAL_FRAMES} cinematic frames …")
        for i, ((cx, cy, cz), (tx, ty, tz)) in enumerate(keyframes):
            await page.evaluate(
                f"window.__setCam({cx:.3f},{cy:.3f},{cz:.3f},{tx:.3f},{ty:.3f},{tz:.3f})"
            )
            # Wait for one rAF so the WebGL frame and CSS compositing both settle.
            await page.evaluate(
                "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
            )
            await page.wait_for_timeout(60)

            raw = await page.screenshot(type="png", full_page=False)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if img.width != GIF_W:
                img = img.resize((GIF_W, GIF_H), Image.LANCZOS)
            rgb_frames.append(img)
            print(f"  frame {i + 1:>2}/{TOTAL_FRAMES}", end="\r")

        await browser.close()

    print(f"Building GIF ({TOTAL_FRAMES} frames, {FPS} fps, 256-colour dithered) …")
    # Per-frame quantisation with dithering gives the best colour accuracy.
    # Using a shared palette caused blank frames when the reference palette
    # was biased toward the first (establish-shot) colours.
    palette_frames: list["Image.Image"] = [
        f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=1)
        for f in rgb_frames
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    palette_frames[0].save(
        out,
        save_all=True,
        append_images=palette_frames[1:],
        optimize=True,
        duration=int(1000 / FPS),
        loop=0,
    )
    size_kb = out.stat().st_size // 1024
    print(f"Saved → {out}  ({size_kb} KB)")
    if size_kb > 6000:
        print("Tip: reduce TOTAL_FRAMES or VIEWPORT_W constants to shrink the file.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture cinematic GIF from Bridge Viewer")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    asyncio.run(capture(args.url, Path(args.out)))


if __name__ == "__main__":
    main()
