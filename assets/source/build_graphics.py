#!/usr/bin/env python3
"""Build José Quiros's city-pop profile graphics from assets/source/avatar.png.

Outputs dark/light animated banners, a whoami card, skill radars, and a few
custom skill tiles. Animations are SMIL only. Photos are embedded as base64
so the SVG still renders inside GitHub's image proxy.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "assets" / "source" / "avatar.png"
OUT = ROOT / "assets"
ICONS = OUT / "icons"

FRAME = 540
CYCLE = 16.0  # seconds, four portrait treatments

THEMES = {
    "dark": {
        "bg": "#0C1020",
        "window": "#12182C",
        "panel": "#161E36",
        "stroke": "#31405F",
        "pink": "#FF7EB3",
        "lav": "#C9B6FF",
        "cyan": "#7EE0F0",
        "gold": "#F6D58A",
        "text": "#F7F0F8",
        "muted": "#93A0BD",
        "dim": "#6D7B98",
        "shadow": "#02040C",
        "shadow_op": "0.45",
        "ink": (255, 126, 179),
        "paper": (22, 30, 54),
        "halo": (255, 126, 179),
        "status_fg": "#0C1020",
        "scan": (0, 0, 0, 70),
    },
    "light": {
        "bg": "#FFF6F8",
        "window": "#FFFFFF",
        "panel": "#FFF0F5",
        "stroke": "#F0C4D2",
        "pink": "#E23E78",
        "lav": "#7A5BD0",
        "cyan": "#0E7490",
        "gold": "#9A6700",
        "text": "#2A2033",
        "muted": "#8C6676",
        "dim": "#B08998",
        "shadow": "#E7A8BA",
        "shadow_op": "0.35",
        "ink": (194, 24, 91),
        "paper": (255, 240, 245),
        "halo": (226, 62, 120),
        "status_fg": "#FFFFFF",
        "scan": (90, 20, 40, 28),
    },
}


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def jpeg_b64(image: Image.Image, quality: int = 84) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_avatar() -> Image.Image:
    image = Image.open(SRC).convert("RGB")
    image = image.resize((FRAME, FRAME), Image.Resampling.LANCZOS)
    return image


def luminance(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722


def stretch(gray: np.ndarray, lo: float = 3.0, hi: float = 97.5) -> np.ndarray:
    a, b = np.percentile(gray, [lo, hi])
    return np.clip((gray - a) / (b - a + 1e-6), 0.0, 1.0)


def vignette(arr: np.ndarray, strength: float = 0.35) -> np.ndarray:
    h, w = arr.shape[:2]
    y = np.linspace(-1.0, 1.0, h)[:, None]
    x = np.linspace(-1.0, 1.0, w)[None, :]
    mask = np.clip(1.0 - strength * (x * x + y * y), 0.0, 1.0)
    return arr * mask[..., None]


def studio_frame(avatar: Image.Image) -> Image.Image:
    arr = np.asarray(avatar, dtype=np.float32) / 255.0
    # Gentle S-curve. Keep the red shirt and the room recognizable.
    arr = np.clip((arr - 0.5) * 1.08 + 0.52, 0.0, 1.0)
    lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
    highlight = np.array([1.0, 0.86, 0.92], dtype=np.float32)
    arr = arr * 0.92 + (highlight * lum[..., None]) * 0.08
    arr = vignette(np.clip(arr, 0.0, 1.0), 0.28)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


def duotone_frame(avatar: Image.Image, theme: str) -> Image.Image:
    gray = stretch(luminance(avatar), 2.0, 98.0)
    gray = np.clip(gray**0.92, 0.0, 1.0)
    if theme == "dark":
        stops = [
            (0.0, np.array([16, 10, 32], dtype=np.float32)),
            (0.42, np.array([255, 90, 160], dtype=np.float32)),
            (0.72, np.array([255, 186, 214], dtype=np.float32)),
            (1.0, np.array([188, 250, 255], dtype=np.float32)),
        ]
    else:
        stops = [
            (0.0, np.array([74, 22, 58], dtype=np.float32)),
            (0.45, np.array([226, 62, 120], dtype=np.float32)),
            (0.75, np.array([255, 196, 214], dtype=np.float32)),
            (1.0, np.array([255, 248, 250], dtype=np.float32)),
        ]
    out = np.zeros(gray.shape + (3,), dtype=np.float32)
    for i in range(len(stops) - 1):
        a_t, a_c = stops[i]
        b_t, b_c = stops[i + 1]
        span = b_t - a_t
        weight = np.clip((gray - a_t) / span, 0.0, 1.0)
        active = (gray >= a_t) & (gray <= b_t if i < len(stops) - 2 else gray <= b_t + 1)
        # Paint the segment, later stops overwrite the boundary.
        seg = a_c + (b_c - a_c) * weight[..., None]
        out = np.where(active[..., None], seg, out)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    work = gray.astype(np.float32).copy()
    h, w = work.shape
    for y in range(h):
        forward = y % 2 == 0
        xs = range(w) if forward else range(w - 1, -1, -1)
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.45 else 0.0
            work[y, x] = new
            err = old - new
            if forward:
                if x + 1 < w:
                    work[y, x + 1] += err * 0.4375
                if y + 1 < h:
                    if x > 0:
                        work[y + 1, x - 1] += err * 0.1875
                    work[y + 1, x] += err * 0.3125
                    if x + 1 < w:
                        work[y + 1, x + 1] += err * 0.0625
            else:
                if x - 1 >= 0:
                    work[y, x - 1] += err * 0.4375
                if y + 1 < h:
                    if x + 1 < w:
                        work[y + 1, x + 1] += err * 0.1875
                    work[y + 1, x] += err * 0.3125
                    if x - 1 >= 0:
                        work[y + 1, x - 1] += err * 0.0625
    return work


def dither_frame(avatar: Image.Image, theme: str) -> Image.Image:
    gray = stretch(luminance(avatar), 6.0, 96.0)
    # Open the face a little before the 1-bit cut.
    gray = np.clip((gray - 0.05) * 1.15, 0.0, 1.0)
    bits = floyd_steinberg(gray)
    palette = THEMES[theme]
    ink = np.array(palette["ink"], dtype=np.float32)
    paper = np.array(palette["paper"], dtype=np.float32)
    out = np.where(bits[..., None] > 0.5, ink, paper)
    return Image.fromarray(out.astype(np.uint8), "RGB")


def halftone_frame(avatar: Image.Image, theme: str) -> Image.Image:
    palette = THEMES[theme]
    gray = stretch(luminance(avatar), 4.0, 97.0)
    paper = palette["paper"]
    ink = palette["ink"]
    image = Image.new("RGB", (FRAME, FRAME), paper)
    draw = ImageDraw.Draw(image)
    cell = 8
    max_r = cell * 0.48
    for y in range(0, FRAME, cell):
        offset = (cell // 2) if (y // cell) % 2 else 0
        for x in range(-cell, FRAME, cell):
            cx = x + cell // 2 + offset
            cy = y + cell // 2
            if not (0 <= cx < FRAME and 0 <= cy < FRAME):
                continue
            x0, x1 = max(0, cx - 2), min(FRAME, cx + 3)
            y0, y1 = max(0, cy - 2), min(FRAME, cy + 3)
            sample = float(gray[y0:y1, x0:x1].mean())
            radius = (1.0 - sample) * max_r
            if radius < 0.7:
                continue
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=ink,
            )
    return image


def build_frames(avatar: Image.Image, theme: str) -> list[tuple[str, str, str]]:
    """Return (label, mime, base64) for the four changing portraits."""
    studio = studio_frame(avatar)
    duo = duotone_frame(avatar, theme)
    dither = dither_frame(avatar, theme)
    half = halftone_frame(avatar, theme)
    frames = [
        ("01  STUDIO", "image/jpeg", jpeg_b64(studio, 86)),
        ("02  DUOTONE", "image/jpeg", jpeg_b64(duo, 86)),
        ("03  1-BIT", "image/png", png_b64(dither)),
        ("04  HALFTONE", "image/jpeg", jpeg_b64(half, 86)),
    ]
    sheet = Image.new("RGB", (FRAME * 4 + 30, FRAME + 8), (12, 16, 32))
    for i, frame in enumerate((studio, duo, dither, half)):
        sheet.paste(frame, (8 + i * (FRAME + 6), 4))
    sheet.save(OUT / f"_preview-{theme}.png")
    return frames


def frame_opacity(index: int) -> tuple[str, str]:
    """SMIL keyTimes/values so frame `index` holds, then crossfades."""
    # Holds: [0,4), [4,8), [8,12), [12,16) with a 0.45s crossfade.
    fade = 0.45 / CYCLE
    hold = 0.25
    start = index * hold
    end = start + hold
    if index == 0:
        keys = [0.0, hold - fade, hold, 1.0 - fade, 1.0]
        vals = [1, 1, 0, 0, 1]
    else:
        keys = [0.0, start, start + fade, end - fade, end, 1.0]
        vals = [0, 0, 1, 1, 0, 0]
    key_s = ";".join(f"{k:.4f}" for k in keys)
    val_s = ";".join(str(v) for v in vals)
    return key_s, val_s


def portrait_layers(frames: list[tuple[str, str, str]], theme: dict) -> str:
    x, y, size = 48, 128, 392
    cx, cy = x + size / 2, y + size / 2
    parts = [
        f'<clipPath id="photoClip"><rect x="{x}" y="{y}" width="{size}" height="{size}" rx="8"/></clipPath>'
    ]
    images = []
    for i, (label, mime, data) in enumerate(frames):
        keys, vals = frame_opacity(i)
        images.append(
            f'<g opacity="{"1" if i == 0 else "0"}">'
            f'<animate attributeName="opacity" dur="{CYCLE}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="{keys}" values="{vals}"/>'
            f'<image x="{x}" y="{y}" width="{size}" height="{size}" '
            f'preserveAspectRatio="xMidYMid slice" '
            f'href="data:{mime};base64,{data}" xlink:href="data:{mime};base64,{data}"/>'
            f"</g>"
        )
    # Scanlines drift across the still.
    scan_lines = []
    for i in range(0, size, 4):
        scan_lines.append(
            f'<rect x="{x}" y="{y + i}" width="{size}" height="1" fill="#000" opacity="0.16"/>'
        )
    beam = (
        f'<rect x="{x}" y="{y}" width="{size}" height="46" fill="url(#beam)" opacity="0.35">'
        f'<animateTransform attributeName="transform" type="translate" '
        f'values="0 -46;0 {size}" dur="3.6s" repeatCount="indefinite" calcMode="linear"/>'
        f"</rect>"
    )
    flash_keys = "0;0.22;0.25;0.28;0.47;0.50;0.53;0.72;0.75;0.78;0.97;1"
    flash_vals = "0;0;0.22;0;0;0.22;0;0;0.22;0;0.22;0"
    flash = (
        f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{theme["pink"]}" opacity="0">'
        f'<animate attributeName="opacity" dur="{CYCLE}s" repeatCount="indefinite" '
        f'calcMode="linear" keyTimes="{flash_keys}" values="{flash_vals}"/>'
        f"</rect>"
    )
    labels = []
    for i, (label, _mime, _data) in enumerate(frames):
        keys, vals = frame_opacity(i)
        labels.append(
            f'<g opacity="{"1" if i == 0 else "0"}">'
            f'<animate attributeName="opacity" dur="{CYCLE}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="{keys}" values="{vals}"/>'
            f'<rect x="{x + 8}" y="{y + size - 34}" width="132" height="22" rx="4" fill="#0C1020" opacity="0.62"/>'
            f'<text x="{x + 16}" y="{y + size - 18}" fill="#F7F0F8" '
            f'font-family="{MONO}" font-size="12" font-weight="700" letter-spacing="1.2">{esc(label)}</text>'
            f"</g>"
        )
    parts.append(
        f'<g clip-path="url(#photoClip)">{"".join(images)}{"".join(scan_lines)}{beam}{flash}{"".join(labels)}</g>'
    )
    # Corner brackets
    b = theme["lav"]
    parts.append(
        f'<path d="M{x} {y+16}V{y}H{x+16}M{x+size-16} {y}H{x+size}V{y+16}'
        f'M{x} {y+size-16}V{y+size}H{x+16}M{x+size-16} {y+size}H{x+size}V{y+size-16}" '
        f'fill="none" stroke="{b}" stroke-width="1.5" opacity="0.85"/>'
    )
    # Four progress pips
    for i in range(4):
        keys, vals = frame_opacity(i)
        px = x + size - 78 + i * 16
        py = y + size - 22
        parts.append(
            f'<circle cx="{px}" cy="{py}" r="3.2" fill="{theme["dim"]}" opacity="0.55"/>'
            f'<circle cx="{px}" cy="{py}" r="3.2" fill="{theme["pink"]}" opacity="{"1" if i == 0 else "0"}">'
            f'<animate attributeName="opacity" dur="{CYCLE}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="{keys}" values="{vals}"/>'
            f"</circle>"
        )
    # Silence unused center (kept for future ken-burns tuning).
    _ = (cx, cy)
    return "".join(parts)


MONO = "DejaVu Sans Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

PROFILE_LINES = [
    ("sec", "profile:"),
    ("kv", "subject", "José Quiros"),
    ("kv", "handle", "@jose-JQ"),
    ("kv", "focus", "Data Science · ML · AI"),
    ("kv", "bi", "ETL · Oracle · PostgreSQL · Power BI"),
    ("kv", "stack", "Python · React · TypeScript · Node.js"),
    ("kv", "security", "ethical hacking"),
    ("kv", "also", "cooking · swimming · games"),
    ("sec", "projects:"),
    ("kv", "manpac", "AI invoice extraction"),
    ("kv", "rag", "multimodal search"),
    ("kv", "ir", "TF-IDF · BM25 · React"),
    ("sec", "contact:"),
    ("kv", "github", "jose-JQ"),
    ("kv", "email", "j26quiros@gmail.com"),
]


def yaml_block(theme: dict) -> str:
    chunks = []
    y0, dy = 156, 22
    for i, line in enumerate(PROFILE_LINES):
        y = y0 + i * dy
        begin = f"{0.12 + i * 0.06:.2f}s"
        num = f"{i + 1:>2}"
        if line[0] == "sec":
            body = (
                f'<text x="542" y="{y}" font-family="{MONO}" font-size="13">'
                f'<tspan fill="{theme["lav"]}" font-weight="700">{esc(line[1])}</tspan></text>'
            )
        else:
            body = (
                f'<text x="562" y="{y}" font-family="{MONO}" font-size="13">'
                f'<tspan fill="{theme["pink"]}">{esc(line[1])}: </tspan>'
                f'<tspan fill="{theme["text"]}">{esc(line[2])}</tspan></text>'
            )
        chunks.append(
            f'<g opacity="0">'
            f'<animate attributeName="opacity" begin="{begin}" dur="0.28s" values="0;1" fill="freeze"/>'
            f'<text x="520" y="{y}" text-anchor="end" fill="{theme["dim"]}" opacity="0.85" '
            f'font-family="{MONO}" font-size="12">{num}</text>'
            f"{body}</g>"
        )
    cursor_y = y0 + len(PROFILE_LINES) * dy - 11
    chunks.append(
        f'<rect x="562" y="{cursor_y}" width="8" height="15" fill="{theme["pink"]}">'
        f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.45;0.5;1" '
        f'dur="1.15s" repeatCount="indefinite"/>'
        f"</rect>"
    )
    return "".join(chunks)


def banner_svg(theme_name: str, frames: list[tuple[str, str, str]]) -> str:
    t = THEMES[theme_name]
    portrait = portrait_layers(frames, t)
    yaml = yaml_block(t)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1180" height="640" viewBox="0 0 1180 640" role="img" aria-labelledby="title desc">
  <title id="title">José Quiros — live profile</title>
  <desc id="desc">City-pop terminal banner. José's portrait cycles through studio, duotone, 1-bit dither and halftone frames beside a profile.yml buffer.</desc>
  <defs>
    <filter id="shadow" x="-8%" y="-8%" width="120%" height="130%">
      <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="{t["shadow"]}" flood-opacity="{t["shadow_op"]}"/>
    </filter>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{t["pink"]}">
        <animate attributeName="stop-color" dur="8s" repeatCount="indefinite" values="{t["pink"]};{t["cyan"]};{t["lav"]};{t["pink"]}"/>
      </stop>
      <stop offset="0.5" stop-color="{t["lav"]}">
        <animate attributeName="stop-color" dur="8s" repeatCount="indefinite" values="{t["lav"]};{t["pink"]};{t["cyan"]};{t["lav"]}"/>
      </stop>
      <stop offset="1" stop-color="{t["cyan"]}">
        <animate attributeName="stop-color" dur="8s" repeatCount="indefinite" values="{t["cyan"]};{t["lav"]};{t["pink"]};{t["cyan"]}"/>
      </stop>
    </linearGradient>
    <linearGradient id="beam" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{t["cyan"]}" stop-opacity="0"/>
      <stop offset="0.5" stop-color="{t["cyan"]}" stop-opacity="0.55"/>
      <stop offset="1" stop-color="{t["cyan"]}" stop-opacity="0"/>
    </linearGradient>
    {portrait.split("</clipPath>")[0]}</clipPath>
  </defs>
  <rect width="1180" height="640" rx="18" fill="{t["bg"]}"/>
  <rect x="14" y="14" width="1152" height="612" rx="14" fill="{t["window"]}" stroke="{t["stroke"]}" filter="url(#shadow)"/>
  <rect x="14" y="14" width="1152" height="4" fill="url(#accent)"/>
  <circle cx="40" cy="44" r="6" fill="#FF5F57"/>
  <circle cx="62" cy="44" r="6" fill="#FEBC2E"/>
  <circle cx="84" cy="44" r="6" fill="#28C840"/>
  <text x="590" y="49" text-anchor="middle" fill="{t["muted"]}" font-family="{MONO}" font-size="13" letter-spacing="0.6">vim profile.yml</text>
  <text x="1128" y="49" text-anchor="end" fill="{t["pink"]}" font-family="{MONO}" font-size="12" font-weight="700">@jose-JQ</text>
  <path d="M14 68H1166" stroke="{t["stroke"]}"/>

  <rect x="32" y="86" width="424" height="520" rx="8" fill="{t["panel"]}" stroke="{t["stroke"]}"/>
  <text x="48" y="114" fill="{t["lav"]}" font-family="{MONO}" font-size="13" font-weight="700" letter-spacing="1.4">VISUAL.MAP</text>
  <circle cx="168" cy="109" r="4" fill="#FF4D6A">
    <animate attributeName="opacity" values="1;0.25;1" dur="1.4s" repeatCount="indefinite"/>
  </circle>
  <text x="178" y="114" fill="{t["pink"]}" font-family="{MONO}" font-size="11" font-weight="700">LIVE</text>
  <text x="440" y="114" text-anchor="end" fill="{t["muted"]}" font-family="{MONO}" font-size="11">4 FRAMES · 16s</text>
  {portrait.split("</clipPath>", 1)[1]}
  <text x="48" y="582" fill="{t["dim"]}" font-family="{MONO}" font-size="10">SRC avatar.png · FS/SERPENTINE · CITY-POP</text>

  <rect x="472" y="86" width="676" height="520" rx="8" fill="{t["panel"]}" stroke="{t["stroke"]}"/>
  <text x="490" y="114" fill="{t["lav"]}" font-family="{MONO}" font-size="13" font-weight="700" letter-spacing="0.8">profile.yml</text>
  <text x="590" y="114" fill="{t["muted"]}" font-family="{MONO}" font-size="11">[YAML]</text>
  <rect x="1004" y="96" width="124" height="24" rx="12" fill="{t["lav"]}" opacity="0.16" stroke="{t["lav"]}"/>
  <text x="1066" y="113" text-anchor="middle" fill="{t["lav"]}" font-family="{MONO}" font-size="12" font-weight="700">jose-JQ</text>
  {yaml}
  <path d="M473 558H1147" stroke="{t["stroke"]}"/>
  <rect x="486" y="570" width="72" height="20" rx="3" fill="{t["pink"]}"/>
  <text x="522" y="584" text-anchor="middle" fill="{t["status_fg"]}" font-family="{MONO}" font-size="11" font-weight="700">NORMAL</text>
  <text x="568" y="584" fill="{t["text"]}" font-family="{MONO}" font-size="12" font-weight="600">profile.yml</text>
  <text x="740" y="584" fill="{t["muted"]}" font-family="{MONO}" font-size="11">[utf-8]</text>
  <text x="1130" y="584" text-anchor="end" fill="{t["muted"]}" font-family="{MONO}" font-size="11">15L · 100%</text>
</svg>
'''


def whoami_svg(avatar_b64: str) -> str:
    t = THEMES["dark"]
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 960 430" width="960" height="430" role="img" aria-labelledby="title desc">
  <title id="title">José Quiros — whoami</title>
  <desc id="desc">City-pop terminal card with José Quiros's focus areas and portrait.</desc>
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="1" y2="1">
      <stop stop-color="{t["pink"]}"/>
      <stop offset="0.5" stop-color="{t["lav"]}"/>
      <stop offset="1" stop-color="{t["cyan"]}"/>
    </linearGradient>
    <clipPath id="face"><circle cx="792" cy="214" r="62"/></clipPath>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="2.2" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="960" height="430" rx="18" fill="{t["bg"]}"/>
  <rect x="10" y="10" width="940" height="410" rx="14" fill="none" stroke="url(#sky)" stroke-width="2"/>
  <circle cx="36" cy="38" r="5.5" fill="#FF5F57"/>
  <circle cx="56" cy="38" r="5.5" fill="#FEBC2E"/>
  <circle cx="76" cy="38" r="5.5" fill="#28C840"/>
  <text x="98" y="43" fill="{t["muted"]}" font-family="{MONO}" font-size="14">jose-JQ  ·  profile shell  ·  online</text>
  <path d="M26 60H934" stroke="{t["stroke"]}"/>

  <rect x="28" y="80" width="600" height="322" rx="10" fill="{t["panel"]}" stroke="{t["lav"]}" stroke-width="1.4"/>
  <text x="48" y="116" fill="{t["cyan"]}" font-family="{MONO}" font-size="20" font-weight="700">❯ whoami</text>
  <rect x="168" y="100" width="9" height="18" fill="{t["pink"]}">
    <animate attributeName="opacity" values="1;0;1" keyTimes="0;0.5;1" dur="1.15s" repeatCount="indefinite"/>
  </rect>
  <text x="48" y="152" fill="{t["text"]}" font-family="{MONO}" font-size="18">jose_quiros</text>
  <text x="196" y="152" fill="{t["dim"]}" font-family="{MONO}" font-size="18">—</text>
  <text x="222" y="152" fill="{t["gold"]}" font-family="{MONO}" font-size="16">Data Science · ML · Full-stack</text>
  <path d="M48 172H596" stroke="{t["stroke"]}"/>
  <text x="48" y="204" fill="{t["muted"]}" font-family="{MONO}" font-size="15">context:</text>
  <text x="168" y="204" fill="{t["lav"]}" font-family="{MONO}" font-size="15">raw data -&gt; actionable intelligence</text>
  <text x="48" y="232" fill="{t["muted"]}" font-family="{MONO}" font-size="15">mission:</text>
  <text x="168" y="232" fill="{t["cyan"]}" font-family="{MONO}" font-size="15">ETL, models, and products that ship</text>
  <text x="48" y="274" fill="{t["cyan"]}" font-family="{MONO}" font-size="18" font-weight="700">❯ ls focus/</text>
  <text x="48" y="308" fill="{t["pink"]}" font-family="{MONO}" font-size="14">data/</text>
  <text x="168" y="308" fill="{t["text"]}" font-family="{MONO}" font-size="14">Data Science · Machine Learning · AI</text>
  <text x="48" y="332" fill="{t["pink"]}" font-family="{MONO}" font-size="14">bi/</text>
  <text x="168" y="332" fill="{t["text"]}" font-family="{MONO}" font-size="14">ETL · Oracle · PostgreSQL · Power BI</text>
  <text x="48" y="356" fill="{t["pink"]}" font-family="{MONO}" font-size="14">stack/</text>
  <text x="168" y="356" fill="{t["text"]}" font-family="{MONO}" font-size="14">Python · React · TypeScript · Node.js</text>
  <text x="48" y="380" fill="{t["pink"]}" font-family="{MONO}" font-size="14">security/</text>
  <text x="168" y="380" fill="{t["text"]}" font-family="{MONO}" font-size="14">cybersecurity · ethical hacking</text>

  <rect x="648" y="80" width="284" height="322" rx="10" fill="{t["panel"]}" stroke="{t["lav"]}" stroke-width="1.4"/>
  <text x="670" y="112" fill="{t["gold"]}" font-family="{MONO}" font-size="15">city-pop runtime</text>
  <circle cx="792" cy="214" r="66" fill="none" stroke="url(#sky)" stroke-width="3"/>
  <g clip-path="url(#face)">
    <image href="data:image/jpeg;base64,{avatar_b64}" xlink:href="data:image/jpeg;base64,{avatar_b64}" x="730" y="152" width="124" height="124" preserveAspectRatio="xMidYMid slice"/>
    <g opacity="0.28">
      <path d="M730 168H854M730 176H854M730 184H854M730 192H854M730 200H854M730 208H854M730 216H854M730 224H854M730 232H854M730 240H854M730 248H854M730 256H854M730 264H854" stroke="#0C1020" stroke-width="2"/>
    </g>
  </g>
  <path d="M670 300C706 274 742 324 778 296S850 270 910 304" fill="none" stroke="{t["cyan"]}" stroke-width="2.4" filter="url(#glow)" stroke-dasharray="7 5">
    <animate attributeName="stroke-dashoffset" values="0;-48" dur="1.6s" repeatCount="indefinite"/>
  </path>
  <path d="M670 318C710 292 746 340 786 310S858 286 910 320" fill="none" stroke="{t["pink"]}" stroke-width="1.8" stroke-dasharray="4 6">
    <animate attributeName="stroke-dashoffset" values="0;40" dur="2.1s" repeatCount="indefinite"/>
  </path>
  <text x="670" y="372" fill="{t["muted"]}" font-family="{MONO}" font-size="13">status: always learning</text>
</svg>
'''


SKILLS = [
    ("Python", 0.90),
    ("Data Science", 0.86),
    ("Machine Learning", 0.82),
    ("Full-stack", 0.76),
    ("BI &amp; ETL", 0.72),
    ("Databases", 0.68),
    ("Security", 0.60),
]


def radar_svg(theme_name: str) -> str:
    t = THEMES[theme_name]
    grid = t["stroke"]
    spoke = t["stroke"]
    accent = t["pink"]
    n = len(SKILLS)
    radius = 168
    cx, cy = 360, 292

    def point(index: float, scale: float) -> tuple[float, float]:
        ang = -np.pi / 2 + index * 2 * np.pi / n
        return cx + np.cos(ang) * radius * scale, cy + np.sin(ang) * radius * scale

    rings = []
    for scale, opacity in ((1, 0.9), (0.75, 0.7), (0.5, 0.55), (0.25, 0.4)):
        pts = " ".join(f"{point(i, scale)[0]:.1f},{point(i, scale)[1]:.1f}" for i in range(n))
        rings.append(
            f'<polygon points="{pts}" fill="none" stroke="{grid}" stroke-width="1" opacity="{opacity}"/>'
        )
    spokes = []
    for i in range(n):
        x, y = point(i, 1)
        spokes.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{spoke}" stroke-width="1" opacity="0.65"/>')
    local = " ".join(
        f"{(point(i, score)[0] - cx):.1f},{(point(i, score)[1] - cy):.1f}"
        for i, (_label, score) in enumerate(SKILLS)
    )
    dots = []
    for i, (_label, score) in enumerate(SKILLS):
        x, y = point(i, score)
        dots.append(
            f'<circle cx="{(x - cx):.1f}" cy="{(y - cy):.1f}" r="4.2" fill="{accent}"/>'
        )
    labels = []
    for i, (label, _score) in enumerate(SKILLS):
        x, y = point(i, 1.28)
        ang = -np.pi / 2 + i * 2 * np.pi / n
        if abs(np.cos(ang)) < 0.28:
            anchor = "middle"
        elif np.cos(ang) > 0:
            anchor = "start"
        else:
            anchor = "end"
        labels.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO}" '
            f'font-size="13" font-weight="600" fill="{t["text"]}">{label}</text>'
        )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 560" width="720" height="560" role="img" aria-label="José Quiros skill radar">
  <rect width="720" height="560" fill="{t["bg"]}"/>
  <text x="360" y="34" text-anchor="middle" font-family="{MONO}" font-size="14" font-weight="700" letter-spacing="1.6" fill="{t["lav"]}">SKILL.RADAR</text>
  {''.join(rings)}
  {''.join(spokes)}
  <g transform="translate({cx} {cy})">
    <g>
      <animateTransform attributeName="transform" type="scale" values="0.04;1" dur="1.1s" calcMode="spline" keyTimes="0;1" keySplines="0.22 1 0.36 1" fill="freeze"/>
      <polygon points="{local}" fill="{accent}" fill-opacity="0.22" stroke="{accent}" stroke-width="2.4" stroke-linejoin="round"/>
      {''.join(dots)}
    </g>
  </g>
  {''.join(labels)}
  <text x="360" y="540" text-anchor="middle" font-family="{MONO}" font-size="11" fill="{t["muted"]}">self-rated · public work · not a benchmark</text>
</svg>
'''


def icon(bg: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 256 256">\n'
        f'  <rect width="256" height="256" rx="60" fill="{bg}"/>\n'
        f"  {body}\n"
        "</svg>\n"
    )


def write_icons() -> None:
    ICONS.mkdir(parents=True, exist_ok=True)
    tiles = {
        "oracle": icon(
            "#C74634",
            '<ellipse cx="128" cy="128" rx="78" ry="34" fill="none" stroke="#fff" stroke-width="22"/>',
        ),
        "powerbi": icon(
            "#F2C811",
            '<rect x="58" y="132" width="36" height="66" rx="6" fill="#1A1A1A"/>'
            '<rect x="110" y="96" width="36" height="102" rx="6" fill="#1A1A1A"/>'
            '<rect x="162" y="58" width="36" height="140" rx="6" fill="#1A1A1A"/>',
        ),
        "pandas": icon(
            "#150458",
            '<rect x="52" y="48" width="152" height="160" rx="12" fill="none" stroke="#fff" stroke-width="10"/>'
            '<path d="M52 96H204M52 144H204M104 48V208M156 48V208" stroke="#E6F4FF" stroke-width="8"/>',
        ),
        "numpy": icon(
            "#013243",
            '<rect x="48" y="48" width="70" height="70" rx="8" fill="#4DABCF"/>'
            '<rect x="138" y="48" width="70" height="70" rx="8" fill="#76D0F1"/>'
            '<rect x="48" y="138" width="70" height="70" rx="8" fill="#76D0F1"/>'
            '<rect x="138" y="138" width="70" height="70" rx="8" fill="#4DABCF"/>',
        ),
        "jupyter": icon(
            "#F37626",
            '<rect x="68" y="40" width="120" height="176" rx="12" fill="#fff"/>'
            '<circle cx="128" cy="104" r="22" fill="#F37626"/>'
            '<path d="M92 150H164M92 174H148" stroke="#F37626" stroke-width="10" stroke-linecap="round"/>',
        ),
        "keras": icon(
            "#D00000",
            '<path d="M78 196V60L178 128L78 196Z" fill="#fff"/>'
            '<path d="M118 128H190" stroke="#fff" stroke-width="16" stroke-linecap="round"/>',
        ),
        "streamlit": icon(
            "#FF4B4B",
            '<path d="M128 44C92 92 70 112 70 146c0 36 26 62 58 62s58-26 58-62c0-22-12-40-28-62 2 16-6 28-20 36-2-28-18-48-10-76z" fill="#fff"/>',
        ),
        "security": icon(
            "#3D2E7C",
            '<path d="M128 40L196 68V128c0 46-30 74-68 88-38-14-68-42-68-88V68L128 40Z" fill="none" stroke="#fff" stroke-width="14" stroke-linejoin="round"/>'
            '<path d="M104 128l16 16 34-40" fill="none" stroke="#7EE0F0" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>',
        ),
    }
    for name, svg in tiles.items():
        (ICONS / f"{name}.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    avatar = load_avatar()
    # Remove stale contact sheets from the asset dir after writing banners.
    for theme in ("dark", "light"):
        frames = build_frames(avatar, theme)
        svg = banner_svg(theme, frames)
        path = OUT / f"banner-{theme}.svg"
        path.write_text(svg, encoding="utf-8")
        print(f"{path.name}: {path.stat().st_size / 1024:.1f} KB")
    face = studio_frame(avatar).resize((280, 280), Image.Resampling.LANCZOS)
    who = OUT / "whoami.svg"
    who.write_text(whoami_svg(jpeg_b64(face, 86)), encoding="utf-8")
    print(f"{who.name}: {who.stat().st_size / 1024:.1f} KB")
    for theme in ("dark", "light"):
        path = OUT / f"radar-{theme}.svg"
        path.write_text(radar_svg(theme), encoding="utf-8")
        print(f"{path.name}: {path.stat().st_size / 1024:.1f} KB")
    write_icons()
    for preview in OUT.glob("_preview-*.png"):
        preview.unlink()


if __name__ == "__main__":
    main()
