#!/usr/bin/env python3
"""
amf.render.generate_bg
======================

AMF v4.0 Manuscript Background Generator
-----------------------------------------

Generates three publication-quality background images for the AMF v4.0
research document:

    cover_bg.png      — Ornate front cover with codex frame, corner
                        ornaments, botanical watermarks, and grid
    body_bg.png       — Minimal interior page with subtle ruling and grid
    backcover_bg.png  — Back cover with centered botanical watermark

APPROACH
--------
Backgrounds are constructed as SVG (Scalable Vector Graphics) and rendered
to PNG via Playwright's headless Chromium browser at 2× device pixel ratio
(effective resolution: 1588 × 2246 pixels at A4 proportions).

SVG is chosen because:
  - Fully resolution-independent
  - No external image assets required
  - Deterministic output (same SVG → same PNG)
  - Easy to version-control (text-based format)

DESIGN RATIONALE
----------------
The visual language blends two reference systems:
  1. Historical manuscript aesthetics (vellum tones, hand-drawn-like grain,
     botanical watermarks suggesting the Voynich herbal folios)
  2. Computational/data aesthetics (grid overlay, precise geometric ornaments)

This reflects the AMF framework's dual nature as both a codicological
study and a computational analysis.

COLOR PALETTE
-------------
All colors use the C dictionary (defined below). The palette is inspired
by aged manuscript materials:
  bg          #F5F1E8   Aged vellum / cream parchment
  ink         #2B2D42   Dark indigo-black (primary text/lines)
  indigo      #3D405B   Softer indigo (secondary frames)
  terracotta  #C46A55   Red ochre (rubric accents)
  gold        #B8956A   Ochre gold (ornamental elements)
  sage        #7C9885   Botanical green (plant watermarks)
  rule        #D4CFC2   Light rule lines
  grid        #E8E2D4   Very light grid

REPRODUCIBILITY
---------------
All SVG paths are deterministic given the constants defined here. No
random elements are used. Output PNGs are bit-identical across runs
on the same Playwright/Chromium version.

Output images are saved to the directory specified as the first CLI
argument (default: /mnt/agents/output/amf_v40_assets).

REQUIREMENTS
------------
    playwright>=1.40 (with chromium: playwright install chromium)

USAGE
-----
    python -m amf.render.generate_bg [output_directory]

    # Or via CLI shorthand:
    amf-render --output /path/to/output/
"""

import os
import sys

from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# Page dimensions — A4 at 96 dpi (standard web/screen A4)
# ---------------------------------------------------------------------------

PAGE_W: int = 794   # pixels width  (≈ 210mm at 96dpi)
PAGE_H: int = 1123  # pixels height (≈ 297mm at 96dpi)

# At device_scale_factor=2, actual PNG output is 1588 × 2246 pixels.

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

C: dict[str, str] = {
    "bg":         "#F5F1E8",   # Aged vellum background
    "ink":        "#2B2D42",   # Primary ink (dark indigo-black)
    "indigo":     "#3D405B",   # Secondary frame color
    "terracotta": "#C46A55",   # Accent dots and rubric marks
    "gold":       "#B8956A",   # Ornamental lines and corner pieces
    "sage":       "#7C9885",   # Botanical watermark strokes
    "rule":       "#D4CFC2",   # Ruling lines
    "grid":       "#E8E2D4",   # Background grid (very subtle)
}

# Margin: distance from page edge to primary frame
M: int = 48  # pixels


# ---------------------------------------------------------------------------
# SVG component generators
# ---------------------------------------------------------------------------

def _manuscript_grain() -> str:
    """
    Generate an SVG filter that adds subtle parchment grain texture.

    Uses feTurbulence (fractal noise) composited onto the background
    via multiply blending at very low opacity. This breaks up the
    perfectly flat digital appearance without adding a visible pattern.

    Filter parameters:
        baseFrequency=0.6   Controls grain scale. Higher = finer grain.
        numOctaves=4        Adds detail at multiple scales.
        stitchTiles=stitch  Prevents seaming if tiled.
        opacity=0.08        Applied via color matrix — very subtle.

    Returns
    -------
    str
        SVG <filter> element markup.
    """
    return (
        '<filter id="grain">'
        # feTurbulence generates fractal noise
        '<feTurbulence type="fractalNoise" baseFrequency="0.6" '
        'numOctaves="4" stitchTiles="stitch" result="n"/>'
        # feColorMatrix desaturates and reduces opacity
        '<feColorMatrix type="matrix" '
        'values="0.3 0 0 0 0  0 0.25 0 0 0  0 0 0.2 0 0  0 0 0 0.08 0" '
        'in="n" result="m"/>'
        # feBlend composites the noise onto the source graphic
        '<feBlend in="SourceGraphic" in2="m" mode="multiply"/>'
        '</filter>'
    )


def _botanical_watermark(
    cx: float,
    cy: float,
    scale: float = 1.0,
    opacity: float = 0.06,
) -> str:
    """
    Generate a stylised botanical watermark element.

    Draws a simplified plant/leaf form using two bezier curve paths
    and three accent circles. The design loosely evokes the botanical
    illustrations found in Voynich folios f1–f66, without reproducing
    any specific folio illustration.

    The watermark uses very low opacity (default 0.06) so it reads as
    a background texture rather than a foreground element.

    Parameters
    ----------
    cx, cy
        Center coordinates of the watermark in SVG user units.
    scale
        Scaling factor. 1.0 produces a watermark approximately
        160px tall at PAGE_W=794.
    opacity
        Overall opacity of the watermark group. 0.06 = 6%.

    Returns
    -------
    str
        SVG group element markup.
    """
    s = scale
    return (
        f'<g opacity="{opacity}" transform="translate({cx},{cy}) scale({s})">'

        # Outer leaf/petal path — primary botanical outline
        f'<path d="M0,-80 C30,-60 50,-30 40,0 C30,30 0,50 -20,40 '
        f'C-40,30 -50,0 -30,-30 C-10,-60 20,-70 0,-80" '
        f'fill="none" stroke="{C["sage"]}" stroke-width="1.5"/>'

        # Inner leaf — smaller overlapping shape for depth
        f'<path d="M0,-50 C15,-35 25,-15 20,5 C15,25 -5,35 -15,25 '
        f'C-25,15 -30,-5 -15,-25 C-5,-40 10,-45 0,-50" '
        f'fill="none" stroke="{C["indigo"]}" stroke-width="1"/>'

        # Accent circles — evoke seed/berry elements in botanical illustration
        f'<circle cx="0" cy="-75" r="4" fill="{C["terracotta"]}" opacity="0.5"/>'   # apex
        f'<circle cx="-25" cy="-20" r="3" fill="{C["gold"]}" opacity="0.4"/>'       # mid-left
        f'<circle cx="20" cy="10" r="2.5" fill="{C["sage"]}" opacity="0.4"/>'       # lower-right

        '</g>'
    )


def _computational_grid(
    x: float,
    y: float,
    w: float,
    h: float,
    spacing: int = 28,
    opacity: float = 0.15,
) -> str:
    """
    Generate a uniform grid of horizontal and vertical lines.

    The grid represents the computational/data analysis aspect of the
    AMF framework — a visual metaphor for systematic analysis and
    structured constraint modelling.

    Parameters
    ----------
    x, y
        Top-left origin of the grid area.
    w, h
        Width and height of the grid area.
    spacing
        Distance between grid lines in pixels.
    opacity
        Line opacity. Default 0.15 — very subtle.

    Returns
    -------
    str
        SVG line elements as a single concatenated string.
    """
    lines = ""

    # Vertical lines (columns)
    i = 0
    while x + i * spacing <= x + w:
        lines += (
            f'<line x1="{x + i * spacing}" y1="{y}" '
            f'x2="{x + i * spacing}" y2="{y + h}" '
            f'stroke="{C["rule"]}" stroke-width="0.4" opacity="{opacity}"/>'
        )
        i += 1

    # Horizontal lines (rows)
    i = 0
    while y + i * spacing <= y + h:
        lines += (
            f'<line x1="{x}" y1="{y + i * spacing}" '
            f'x2="{x + w}" y2="{y + i * spacing}" '
            f'stroke="{C["rule"]}" stroke-width="0.4" opacity="{opacity}"/>'
        )
        i += 1

    return lines


def _codex_frame(
    x: float,
    y: float,
    w: float,
    h: float,
    outer_sw: float = 1.8,
    inner_sw: float = 0.6,
    gap: int = 8,
    op: float = 0.7,
) -> str:
    """
    Generate a double-rectangle codex frame (outer + inner border).

    The double-frame motif is common in medieval manuscript page design.
    The outer rectangle is heavier and more opaque; the inner rectangle
    is finer and lighter, creating visual depth.

    Parameters
    ----------
    x, y
        Top-left coordinates of the outer rectangle.
    w, h
        Width and height of the outer rectangle.
    outer_sw
        Stroke width of the outer frame.
    inner_sw
        Stroke width of the inner frame.
    gap
        Inset distance between outer and inner frames in pixels.
    op
        Base opacity for the outer frame. Inner frame is op * 0.6.

    Returns
    -------
    str
        Two SVG <rect> elements.
    """
    return (
        # Outer frame
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="none" stroke="{C["indigo"]}" '
        f'stroke-width="{outer_sw}" opacity="{op}"/>'

        # Inner frame (inset by gap)
        f'<rect x="{x + gap}" y="{y + gap}" '
        f'width="{w - 2*gap}" height="{h - 2*gap}" '
        f'fill="none" stroke="{C["indigo"]}" '
        f'stroke-width="{inner_sw}" opacity="{op * 0.6}"/>'
    )


def _corner_ornaments(
    x: float,
    y: float,
    w: float,
    h: float,
    arm: int = 24,
    sw: float = 1.2,
    op: float = 0.5,
) -> str:
    """
    Generate corner ornaments for a rectangular frame.

    Each corner receives:
    1. An L-shaped bracket (two lines meeting at the corner)
    2. A small diamond accent at the corner point

    The L-bracket arm length controls how far the ornament extends
    along each side. The diamond is a common decorative element in
    historical manuscript borders.

    Parameters
    ----------
    x, y
        Top-left coordinates of the bounding rectangle.
    w, h
        Width and height.
    arm
        Length of the bracket arm in pixels.
    sw
        Stroke width of the bracket lines.
    op
        Opacity of the ornaments.

    Returns
    -------
    str
        SVG path and polygon elements for all four corners.
    """
    # Corner positions with directional multipliers for mirroring
    # (cx, cy, horizontal_dir, vertical_dir)
    corners = [
        (x,     y,     +1, +1),   # top-left
        (x + w, y,     -1, +1),   # top-right
        (x,     y + h, +1, -1),   # bottom-left
        (x + w, y + h, -1, -1),   # bottom-right
    ]

    out = ""
    for cx, cy, dx, dy in corners:
        # L-shaped bracket: vertical arm then horizontal arm
        out += (
            f'<path d="M{cx},{cy + dy*arm} L{cx},{cy} L{cx + dx*arm},{cy}" '
            f'fill="none" stroke="{C["gold"]}" '
            f'stroke-width="{sw}" opacity="{op}"/>'
        )
        # Diamond accent at the corner point
        out += (
            f'<polygon points="'
            f'{cx + dx*5},{cy} '     # right point
            f'{cx},{cy + dy*5} '     # bottom point
            f'{cx - dx*5},{cy} '     # left point
            f'{cx},{cy - dy*5}" '    # top point
            f'fill="{C["gold"]}" opacity="{op * 0.5}"/>'
        )

    return out


def _svg_template(body_content: str) -> str:
    """
    Wrap SVG body content in a complete HTML page for Playwright rendering.

    The HTML wrapper sets the page background to the vellum color and
    resets all margin/padding. The SVG uses 100% width/height to fill
    the page exactly.

    The grain filter is applied as the final layer (rendered on top)
    at 12% opacity, ensuring it visually unifies all elements below.

    Parameters
    ----------
    body_content
        SVG inner content (shapes, paths, etc.) as a string.

    Returns
    -------
    str
        Complete HTML document string, ready for Playwright page.set_content().
    """
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<style>*{{margin:0;padding:0}}'
        f'body{{width:{PAGE_W}px;height:{PAGE_H}px;background:{C["bg"]}}}'
        f'</style></head><body>'
        f'<svg width="{PAGE_W}" height="{PAGE_H}" xmlns="http://www.w3.org/2000/svg">'
        f'<defs>{_manuscript_grain()}</defs>'
        # Base background fill
        f'<rect width="100%" height="100%" fill="{C["bg"]}"/>'
        # All body content (frames, grids, watermarks, etc.)
        f'{body_content}'
        # Grain overlay as top layer — must be last to affect all elements
        f'<rect width="100%" height="100%" filter="url(#grain)" opacity="0.12"/>'
        f'</svg></body></html>'
    )


# ---------------------------------------------------------------------------
# Page layout constants (computed from PAGE_W, PAGE_H, M)
# ---------------------------------------------------------------------------

W, H = PAGE_W, PAGE_H
cx = W // 2    # horizontal center
cy = H // 2    # vertical center

# Cover frame bounds
fx, fy = M, M
fw, fh = W - 2*M, H - 2*M


# ---------------------------------------------------------------------------
# Page body SVG content definitions
# ---------------------------------------------------------------------------

COVER_BODY: str = f"""
{_codex_frame(fx, fy, fw, fh)}
{_corner_ornaments(fx, fy, fw, fh, arm=30, sw=1.4, op=0.6)}
{_computational_grid(fx + 20, fy + 20, fw - 40, fh - 40, spacing=24, opacity=0.12)}
{_botanical_watermark(cx, 260, scale=1.2, opacity=0.07)}
{_botanical_watermark(cx - 180, 700, scale=0.8, opacity=0.05)}
{_botanical_watermark(cx + 200, 850, scale=0.7, opacity=0.05)}
<line x1="{fx + 60}" y1="240" x2="{fx + fw - 60}" y2="240"
 stroke="{C['gold']}" stroke-width="0.8" opacity="0.4"/>
<line x1="{cx - 120}" y1="1008" x2="{cx + 120}" y2="1008"
 stroke="{C['gold']}" stroke-width="0.7" opacity="0.5"/>
<polygon points="{cx},{1022} {cx+5},{1029} {cx},{1036} {cx-5},{1029}"
 fill="{C['gold']}" opacity="0.45"/>
<circle cx="{cx - 18}" cy="1029" r="1.5" fill="{C['terracotta']}" opacity="0.35"/>
<circle cx="{cx + 18}" cy="1029" r="1.5" fill="{C['terracotta']}" opacity="0.35"/>
<circle cx="{fx + 25}" cy="{cy}" r="2" fill="{C['sage']}" opacity="0.25"/>
<circle cx="{fx + fw - 25}" cy="{cy}" r="2" fill="{C['sage']}" opacity="0.25"/>
"""
# Cover notes:
#   - Three botanical watermarks at different scales/positions (large center-top,
#     medium left-mid, small right-lower) create visual asymmetry and depth
#   - Two horizontal rules frame the title area (y=240) and footer (y=1008)
#   - Diamond + flanking circles at y=1029 form a minimal colophon ornament
#   - Two sage circles at mid-left and mid-right of inner frame edge act as
#     subtle registration marks

bx, by = M + 12, M - 4
bw, bh = W - 2*M - 24, H - 2*M + 8

BODY_BODY: str = f"""
<rect x="{bx}" y="{by}" width="{bw}" height="{bh}"
 fill="none" stroke="{C['rule']}" stroke-width="0.5" opacity="0.6"/>
{_corner_ornaments(bx, by, bw, bh, arm=12, sw=0.9, op=0.3)}
{_computational_grid(bx + 2, by + 2, bw - 4, bh - 4, spacing=32, opacity=0.08)}
<line x1="{cx - 60}" y1="{by + bh - 22}" x2="{cx + 60}" y2="{by + bh - 22}"
 stroke="{C['gold']}" stroke-width="0.5" opacity="0.35"/>
"""
# Body page notes:
#   - Lighter frame (rule color, 50% opacity) vs cover (indigo, 70%)
#   - Subtler corner ornaments (arm=12 vs 30, opacity 0.3 vs 0.6)
#   - Wider grid spacing (32px vs 24px) and lower opacity (8% vs 12%)
#   - Single folio-foot rule centered at page bottom

BACK_BODY: str = f"""
{_codex_frame(fx, fy, fw, fh, outer_sw=1.4, inner_sw=0.5, gap=6, op=0.55)}
{_corner_ornaments(fx, fy, fw, fh, arm=22, sw=1.1, op=0.45)}
{_computational_grid(fx + 30, fy + 30, fw - 60, fh - 60, spacing=28, opacity=0.10)}
{_botanical_watermark(cx, cy - 40, scale=0.9, opacity=0.06)}
<line x1="{cx - 80}" y1="{cy + 60}" x2="{cx + 80}" y2="{cy + 60}"
 stroke="{C['gold']}" stroke-width="0.7" opacity="0.4"/>
<polygon points="{cx},{cy + 53} {cx + 4},{cy + 60} {cx},{cy + 67} {cx - 4},{cy + 60}"
 fill="{C['gold']}" opacity="0.35"/>
"""
# Back cover notes:
#   - Slightly lighter than front cover (op=0.55 vs 0.7, arm=22 vs 30)
#   - Single botanical watermark centered on the page (scale 0.9, opacity 6%)
#   - Horizontal rule and diamond ornament at vertical center for visual balance


# ---------------------------------------------------------------------------
# Main rendering function
# ---------------------------------------------------------------------------

def render_backgrounds(output_dir: str) -> None:
    """
    Render all three background images to PNG files.

    Uses Playwright's sync API to launch a headless Chromium instance,
    load each SVG as HTML, and take a screenshot.

    Parameters
    ----------
    output_dir
        Directory to save cover_bg.png, body_bg.png, backcover_bg.png.
        Created if it does not exist.

    Notes
    -----
    device_scale_factor=2 produces physical pixels at 2× the CSS pixel
    size, giving a 1588×2246 PNG from a 794×1123 CSS layout.

    The Chromium process is launched fresh and closed cleanly after
    all three images are rendered (single browser session for efficiency).
    """
    os.makedirs(output_dir, exist_ok=True)

    templates: dict[str, str] = {
        "cover_bg.png":     _svg_template(COVER_BODY),
        "body_bg.png":      _svg_template(BODY_BODY),
        "backcover_bg.png": _svg_template(BACK_BODY),
    }

    with sync_playwright() as p:
        # Launch headless Chromium (no visible window)
        browser = p.chromium.launch()

        # Set viewport to exact page dimensions; scale factor doubles resolution
        page = browser.new_page(
            viewport={"width": PAGE_W, "height": PAGE_H},
            device_scale_factor=2,
        )

        for filename, html_content in templates.items():
            # Load the HTML+SVG into the headless page
            page.set_content(html_content)

            # Capture PNG screenshot of the full page
            output_path = os.path.join(output_dir, filename)
            page.screenshot(path=output_path, type="png")

            print(f"  Rendered: {filename}")

        browser.close()

    print(f"\nDone — AMF v4.0 manuscript backgrounds saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point.

    Usage:
        python -m amf.render.generate_bg [output_directory]

    Default output directory: /mnt/agents/output/amf_v40_assets
    """
    output_dir = (
        sys.argv[1] if len(sys.argv) > 1
        else "/mnt/agents/output/amf_v40_assets"
    )
    render_backgrounds(output_dir)


if __name__ == "__main__":
    main()
