"""Draw the three Design-chapter figures for the report from matplotlib.

Figures produced (200 dpi, white background, about 16 cm wide):
  fig3_1_architecture.png              system architecture
  fig3_2_data_and_fusion_training.png  data split and fusion training
  fig3_3_decision_policy.png           decision policy flowchart

The content is drawn from the delivered code (policy.py, orchestrator.py,
fusion.py, api.py, validation.py, config.py, adapters/image_model.py) and
from the evaluation output files (split_manifest.json, dataset_summary.csv,
fitted/fusion.json), not from prose alone. Where the two disagreed the code
and data files were treated as ground truth.

Colour palette: Okabe-Ito (colour-blind safe) for the branch/module boxes,
and the outcome colours actually used in the web interface stylesheet
(static/style.css: --high, --review, --limited, --unavailable) for the four
terminal outcomes in Figure 3.3.

Run with: python draw_design_figures.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.font_manager import FontProperties

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
# Okabe-Ito colour-blind-safe palette, used consistently for the same
# concept across Figures 3.1 and 3.2.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
}

COLOR_IO = "#EFEFEF"          # input / interface / output boxes
COLOR_VALIDATION = OKABE_ITO["sky_blue"]
COLOR_ORCHESTRATOR = "#DCDCDC"
COLOR_URL = OKABE_ITO["blue"]
COLOR_TEXT = OKABE_ITO["orange"]
COLOR_IMAGE = OKABE_ITO["bluish_green"]
COLOR_EVIDENCE = "#DCDCDC"
COLOR_FUSION = OKABE_ITO["reddish_purple"]
COLOR_POLICY = OKABE_ITO["vermillion"]
COLOR_EXPLANATION = "#DCDCDC"

# Outcome colours, taken from src/phishlens/static/style.css
# (:root custom properties --high, --review, --limited, --unavailable), so
# that Figure 3.3 matches the actual web interface.
COLOR_HIGH = "#e5484d"
COLOR_REVIEW = "#d9a441"
COLOR_LIMITED = "#3fb27f"
COLOR_UNAVAILABLE = "#6b7280"

EDGE_COLOR = "#333333"

CM_TO_IN = 1.0 / 2.54
FIG_WIDTH_CM = 16.0
DPI = 200

FONT_BODY = 9.0
FONT_SMALL = 8.0
FONT_LABEL = 8.5


def _text_color_for(bg_hex: str) -> str:
    """Pick black or white text for readability against a fill colour."""
    bg_hex = bg_hex.lstrip("#")
    r, g, b = (int(bg_hex[i:i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111111" if luminance > 150 else "#ffffff"


def draw_box(ax, x, y, w, h, text, facecolor, fontsize=FONT_BODY, edgecolor=EDGE_COLOR,
             textcolor=None, wrap=26, bold_first_line=False, linewidth=1.1, zorder=3):
    """Draw a rounded rectangle centred at (x, y) with wrapped text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, zorder=zorder,
    )
    ax.add_patch(box)
    if textcolor is None:
        textcolor = _text_color_for(facecolor) if facecolor not in ("none", "white", "#FFFFFF") else "#111111"
    lines = text.split("\n")
    out_lines = []
    for line in lines:
        if line == "":
            out_lines.append("")
        else:
            out_lines.extend(textwrap.wrap(line, width=wrap) or [""])
    ax.text(
        x, y, "\n".join(out_lines), ha="center", va="center",
        fontsize=fontsize, color=textcolor, zorder=zorder + 1, linespacing=1.35,
    )
    return box


def draw_arrow(ax, xy_from, xy_to, color=EDGE_COLOR, style="-|>", lw=1.2,
               connectionstyle="arc3,rad=0.0", shrinkA=2, shrinkB=2, zorder=2):
    arrow = FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=11, linewidth=lw,
        color=color, connectionstyle=connectionstyle, shrinkA=shrinkA, shrinkB=shrinkB,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def label_edge(ax, x, y, text, fontsize=FONT_SMALL, color="#222222", ha="center", style="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color,
             style=style, zorder=4,
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))


def new_axes(width_cm, height_cm):
    fig = plt.figure(figsize=(width_cm * CM_TO_IN, height_cm * CM_TO_IN), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, width_cm)
    ax.set_ylim(0, height_cm)
    ax.set_facecolor("white")
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------
# Figure 3.1: architecture
# --------------------------------------------------------------------------
def draw_figure_3_1(out_path: Path) -> None:
    W = FIG_WIDTH_CM
    cx = W / 2

    # Box half-heights (data coordinates, cm), used to compute positions
    # with a guaranteed minimum edge-to-edge gap between connected boxes.
    h_input = 1.7
    h_validation = 2.3
    h_orchestrator = 1.7
    branch_w = 4.1
    branch_h = 2.6
    h_evidence = 1.9
    h_fusion = 1.5
    h_policy = 1.5
    h_explain = 1.5
    h_output = 2.6

    gap = 0.8          # clear edge-to-edge gap for directly-stacked boxes
    below_row_margin = 1.0  # extra clearance below the adapter row for elbow routing
    top_margin = 0.5

    y_input = 0.0  # placeholder, recomputed below top-down
    # Build positions top-down from a running cursor, each step leaving
    # `gap` of clear space between the bottom edge of the box above and
    # the top edge of the box below.
    cursor = top_margin + h_input / 2
    y_input = cursor
    cursor += h_input / 2 + gap + h_validation / 2
    y_validation = cursor
    cursor += h_validation / 2 + gap + h_orchestrator / 2
    y_orchestrator = cursor
    cursor += h_orchestrator / 2 + gap + branch_h / 2
    y_branches = cursor
    cursor += branch_h / 2 + below_row_margin + gap + h_evidence / 2
    y_evidence = cursor
    cursor += h_evidence / 2 + gap + h_fusion / 2
    y_fusion = cursor
    cursor += h_fusion / 2 + gap + h_policy / 2
    y_policy = cursor
    cursor += h_policy / 2 + gap + h_explain / 2
    y_explain = cursor
    cursor += h_explain / 2 + gap + h_output / 2
    y_output = cursor
    content_bottom = y_output + h_output / 2

    bottom_margin = 0.45
    H = content_bottom + bottom_margin
    # Positions above were measured downward from the top margin; flip to
    # the bottom-up axis coordinate system used by new_axes().
    def flip(y):
        return H - y

    fig, ax = new_axes(W, H)

    y_input = flip(y_input)
    y_validation = flip(y_validation)
    y_orchestrator = flip(y_orchestrator)
    y_branches = flip(y_branches)
    y_evidence = flip(y_evidence)
    y_fusion = flip(y_fusion)
    y_policy = flip(y_policy)
    y_explain = flip(y_explain)
    y_output = flip(y_output)

    draw_box(ax, cx, y_input, 13.4, h_input,
             "User input: URL, message or page text, screenshot\n(at least one input is required)",
             COLOR_IO, fontsize=FONT_BODY, wrap=44)

    draw_box(ax, cx, y_validation, 13.4, h_validation,
             "Validation\nURL: up to 2,048 characters\nText: up to 10,000 characters\nImage: PNG or JPEG, up to 5 MB",
             COLOR_VALIDATION, fontsize=FONT_BODY, wrap=48)

    draw_box(ax, cx, y_orchestrator, 13.4, h_orchestrator,
             "Orchestrator\nruns each provided input through its adapter",
             COLOR_ORCHESTRATOR, fontsize=FONT_BODY, wrap=44)

    x_url = cx - 4.8
    x_text = cx
    x_image = cx + 4.8

    draw_box(ax, x_url, y_branches, branch_w, branch_h,
             "URL adapter\nurlbert-tiny-v4\n(never fetched, R8)",
             COLOR_URL, fontsize=FONT_LABEL, wrap=18)
    draw_box(ax, x_text, y_branches, branch_w, branch_h,
             "Text adapter\nbert-finetuned-\nphishing",
             COLOR_TEXT, fontsize=FONT_LABEL, wrap=18)
    draw_box(ax, x_image, y_branches, branch_w, branch_h,
             "Image adapter\nCLIP ViT-B/32 +\nfitted logistic head",
             COLOR_IMAGE, fontsize=FONT_LABEL, wrap=18)

    draw_box(ax, cx, y_evidence, 13.4, h_evidence,
             "Evidence records (one per branch)\nstatus: ok / unavailable / failed;\na failed branch does not stop the others (R4)",
             COLOR_EVIDENCE, fontsize=FONT_BODY, wrap=52)

    draw_box(ax, cx, y_fusion, 13.4, h_fusion,
             "Fusion module\nlearned / mean / max / best single",
             COLOR_FUSION, fontsize=FONT_BODY, wrap=44)

    draw_box(ax, cx, y_policy, 13.4, h_policy,
             "Decision policy\nthresholds and disagreement gate",
             COLOR_POLICY, fontsize=FONT_BODY, wrap=44)

    draw_box(ax, cx, y_explain, 13.4, h_explain,
             "Explanation builder\ntemplate-based, built only from evidence values",
             COLOR_EXPLANATION, fontsize=FONT_BODY, wrap=50)

    draw_box(ax, cx, y_output, 13.4, h_output,
             "Web interface / JSON result\nGET /\nPOST /api/analyze\nGET /api/health",
             COLOR_IO, fontsize=FONT_BODY, wrap=48)

    # main vertical arrows
    draw_arrow(ax, (cx, y_input - h_input / 2), (cx, y_validation + h_validation / 2))
    draw_arrow(ax, (cx, y_validation - h_validation / 2), (cx, y_orchestrator + h_orchestrator / 2))

    # orchestrator -> three branches
    top_branch = y_branches + branch_h / 2
    draw_arrow(ax, (x_url, y_orchestrator - h_orchestrator / 2), (x_url, top_branch), connectionstyle="arc3,rad=0.0")
    draw_arrow(ax, (cx, y_orchestrator - h_orchestrator / 2), (x_text, top_branch))
    draw_arrow(ax, (x_image, y_orchestrator - h_orchestrator / 2), (x_image, top_branch), connectionstyle="arc3,rad=0.0")

    # three branches -> evidence records, routed as a clean elbow: a plain
    # line straight down from the bottom-centre of the URL/image adapter
    # box to a point clearly below the whole adapter row, then a single
    # arrowhead on the diagonal/straight segment into the evidence box.
    # This keeps the diagonal segment below the adapter row so it never
    # crosses the text adapter box.
    bottom_branch = y_branches - branch_h / 2
    elbow_y = bottom_branch - below_row_margin * 0.55
    top_evidence = y_evidence + h_evidence / 2

    draw_arrow(ax, (x_url, bottom_branch), (x_url, elbow_y), style="-", lw=1.2)
    draw_arrow(ax, (x_url, elbow_y), (cx - 0.05, top_evidence), connectionstyle="arc3,rad=-0.15")

    draw_arrow(ax, (x_text, bottom_branch), (cx, top_evidence))

    draw_arrow(ax, (x_image, bottom_branch), (x_image, elbow_y), style="-", lw=1.2)
    draw_arrow(ax, (x_image, elbow_y), (cx + 0.05, top_evidence), connectionstyle="arc3,rad=0.15")

    draw_arrow(ax, (cx, y_evidence - h_evidence / 2), (cx, y_fusion + h_fusion / 2))
    draw_arrow(ax, (cx, y_fusion - h_fusion / 2), (cx, y_policy + h_policy / 2))
    draw_arrow(ax, (cx, y_policy - h_policy / 2), (cx, y_explain + h_explain / 2))
    draw_arrow(ax, (cx, y_explain - h_explain / 2), (cx, y_output + h_output / 2))

    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3.2: data split and fusion training
# --------------------------------------------------------------------------
def draw_figure_3_2(out_path: Path) -> None:
    W = FIG_WIDTH_CM
    cx = W / 2

    h1 = 1.5
    h2 = 1.5
    h3 = 1.5
    h4 = 1.5
    h5 = 1.7
    h_part = 1.9      # development / locked test partition boxes
    h6 = 2.6          # training / held-out boxes
    h7 = 1.9
    h8 = 1.5

    gap = 0.8
    below_row_margin = 1.0
    top_margin = 0.5

    cursor = top_margin + h1 / 2
    y1 = cursor
    cursor += h1 / 2 + gap + h2 / 2
    y2 = cursor
    cursor += h2 / 2 + gap + h3 / 2
    y3 = cursor
    cursor += h3 / 2 + gap + h4 / 2
    y4 = cursor
    cursor += h4 / 2 + gap + h5 / 2
    y5 = cursor
    cursor += h5 / 2 + gap + h_part / 2
    y_dev = y_test = cursor
    cursor += h_part / 2 + gap + h6 / 2
    y6a = y6b = cursor
    cursor += h6 / 2 + below_row_margin + gap + h7 / 2
    y7 = cursor
    cursor += h7 / 2 + gap + h8 / 2
    y8 = cursor
    content_bottom = y8 + h8 / 2

    bottom_margin = 0.45
    H = content_bottom + bottom_margin

    def flip(y):
        return H - y

    fig, ax = new_axes(W, H)

    y1, y2, y3, y4, y5 = flip(y1), flip(y2), flip(y3), flip(y4), flip(y5)
    y_dev = y_test = flip(y_dev)
    y6a = y6b = flip(y6a)
    y7 = flip(y7)
    y8 = flip(y8)

    x_dev = cx - 4.4
    x_test = cx + 4.4

    draw_box(ax, cx, y1, 13.6, h1,
             "Screenshots fetched: 651 phishing, 743 legitimate\n(last id range of each class)",
             COLOR_IO, fontsize=FONT_BODY, wrap=52)

    draw_box(ax, cx, y2, 13.6, h2,
             "Readable and not blank: 512 phishing, 510 legitimate",
             "#DCDCDC", fontsize=FONT_BODY, wrap=52)

    draw_box(ax, cx, y3, 13.6, h3,
             "Removed for training-set overlap with ealvaradob/\nphishing-dataset: 8 phishing, 21 legitimate",
             "#DCDCDC", fontsize=FONT_BODY, wrap=54)

    draw_box(ax, cx, y4, 13.6, h4,
             "Final linked cases: 993 (504 phishing, 489 legitimate)",
             "#DCDCDC", fontsize=FONT_BODY, wrap=52)

    draw_box(ax, cx, y5, 13.6, h5,
             "Grouped by registered domain of the case URL:\n986 domains, no domain in more than one partition",
             COLOR_VALIDATION, fontsize=FONT_BODY, wrap=54)

    draw_box(ax, x_dev, y_dev, 6.1, h_part,
             "Development partition\n694 cases (352 phishing, 342 legitimate)\n691 domains",
             OKABE_ITO["sky_blue"], fontsize=FONT_LABEL, wrap=32)
    draw_box(ax, x_test, y_test, 6.1, h_part,
             "Locked test partition\n299 cases (152 phishing, 147 legitimate)\n295 domains",
             OKABE_ITO["grey"], fontsize=FONT_LABEL, wrap=32)

    draw_box(ax, x_dev, y6a, 6.4, h6,
             "On development data only:\ngrouped 5-fold CV image head\n(logistic on CLIP embedding);\nlearned fusion (6 features,\nmodality masking, C=0.1);\nPlatt calibration; thresholds",
             OKABE_ITO["reddish_purple"], fontsize=FONT_LABEL, wrap=30)

    draw_box(ax, x_test, y6b, 6.4, h6,
             "Held out, untouched\nuntil scoring\n(no model, threshold or\nprompt choice made\nusing this partition)",
             "#EFEFEF", fontsize=FONT_LABEL, wrap=26)

    draw_box(ax, cx, y7, 13.6, h7,
             "fusion.json frozen: low 0.3519, high 0.7633,\ndisagreement gap 0.9967 (image_head.json frozen likewise)",
             COLOR_FUSION, fontsize=FONT_BODY, wrap=54)

    draw_box(ax, cx, y8, 13.6, h8,
             "Locked test partition scored once, unchanged (Chapter 5)",
             COLOR_POLICY, fontsize=FONT_BODY, wrap=50)

    draw_arrow(ax, (cx, y1 - h1 / 2), (cx, y2 + h2 / 2))
    draw_arrow(ax, (cx, y2 - h2 / 2), (cx, y3 + h3 / 2))
    draw_arrow(ax, (cx, y3 - h3 / 2), (cx, y4 + h4 / 2))
    draw_arrow(ax, (cx, y4 - h4 / 2), (cx, y5 + h5 / 2))

    draw_arrow(ax, (cx - 0.2, y5 - h5 / 2), (x_dev, y_dev + h_part / 2), connectionstyle="arc3,rad=-0.15")
    draw_arrow(ax, (cx + 0.2, y5 - h5 / 2), (x_test, y_test + h_part / 2), connectionstyle="arc3,rad=0.15")
    label_edge(ax, cx, y5 - h5 / 2 - 0.2, "seed 2026, ~30% of each class to test",
               fontsize=FONT_SMALL, ha="center")

    draw_arrow(ax, (x_dev, y_dev - h_part / 2), (x_dev, y6a + h6 / 2))
    draw_arrow(ax, (x_test, y_test - h_part / 2), (x_test, y6b + h6 / 2))

    # Partition-outcome boxes below y6a/y6b to y7/y8: route as a clean
    # elbow (straight down, plain line, then a single arrowhead on the
    # final diagonal) so the diagonal segment stays below the training
    # row and does not cross the neighbouring box.
    elbow_y2 = y6a - h6 / 2 - below_row_margin * 0.55
    top7 = y7 + h7 / 2

    draw_arrow(ax, (x_dev, y6a - h6 / 2), (x_dev, elbow_y2), style="-", lw=1.2)
    draw_arrow(ax, (x_dev, elbow_y2), (cx - 0.15, top7), connectionstyle="arc3,rad=-0.12")

    # The "held out" box connects to the scoring box (y8), which sits
    # below the fusion.json box (y7). Route well clear of y7's footprint:
    # straight down, then out to the right of y7/y8's full width, then
    # straight down again, then a short final segment with the single
    # arrowhead into y8's right edge.
    right_x = cx + 13.6 / 2 + 0.5
    draw_arrow(ax, (x_test, y6b - h6 / 2), (x_test, elbow_y2), style="-", lw=1.2)
    draw_arrow(ax, (x_test, elbow_y2), (right_x, elbow_y2), style="-", lw=1.2)
    draw_arrow(ax, (right_x, elbow_y2), (right_x, y8), style="-", lw=1.2)
    draw_arrow(ax, (right_x, y8), (cx + 13.6 / 2 + 0.05, y8))

    draw_arrow(ax, (cx, y7 - h7 / 2), (cx, y8 + h8 / 2))

    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3.3: decision policy
# --------------------------------------------------------------------------
def draw_figure_3_3(out_path: Path) -> None:
    W = FIG_WIDTH_CM

    # Vertical spine that the rule (condition) boxes sit on, with the
    # terminal outcome boxes offset to the right so the "yes" arrow and
    # label have clear room. Margins are kept so nothing is clipped.
    margin = 0.5
    gate_w = 7.4
    outcome_w = 4.2
    gap_x = 1.6
    group_w = gate_w + gap_x + outcome_w
    group_left = (W - group_w) / 2
    spine_x = group_left + gate_w / 2
    outcome_x = group_left + gate_w + gap_x + outcome_w / 2
    start_w = min(9.6, 2 * (spine_x - margin))

    h_start = 2.0
    h_gate1 = 1.7
    h_gate2 = 2.1
    h_gate3 = 1.7
    h_gate4 = 1.7
    h_rule5 = 1.9

    row_gap = 0.8
    top_margin = 0.5

    cursor = top_margin + h_start / 2
    y_start = cursor
    cursor += h_start / 2 + row_gap + h_gate1 / 2
    y_gate1 = cursor
    cursor += h_gate1 / 2 + row_gap + h_gate2 / 2
    y_gate2 = cursor
    cursor += h_gate2 / 2 + row_gap + h_gate3 / 2
    y_gate3 = cursor
    cursor += h_gate3 / 2 + row_gap + h_gate4 / 2
    y_gate4 = cursor
    cursor += h_gate4 / 2 + row_gap + h_rule5 / 2
    y_rule5 = cursor
    content_bottom = y_rule5 + h_rule5 / 2

    bottom_margin = 0.45
    H = content_bottom + bottom_margin

    def flip(y):
        return H - y

    fig, ax = new_axes(W, H)

    y_start = flip(y_start)
    y_gate1 = flip(y_gate1)
    y_gate2 = flip(y_gate2)
    y_gate3 = flip(y_gate3)
    y_gate4 = flip(y_gate4)
    y_rule5 = flip(y_rule5)

    draw_box(ax, spine_x, y_start, start_w, 2.0,
             "Fused score and the usable branch probabilities\n(branch order: url, text, image)",
             COLOR_IO, fontsize=FONT_BODY, wrap=30)

    draw_box(ax, spine_x, y_gate1, gate_w, 1.7,
             "Rule 1: no usable branch score,\nor fused score is None?",
             "#DCDCDC", fontsize=FONT_BODY, wrap=30)
    draw_box(ax, outcome_x, y_gate1, outcome_w, 1.7,
             "Analysis unavailable",
             COLOR_UNAVAILABLE, fontsize=FONT_BODY, wrap=15)

    draw_box(ax, spine_x, y_gate2, gate_w, 2.1,
             "Rule 2, disagreement gate: 2 or\nmore usable branches, and max\nminus min branch probability\n> 0.9967?",
             "#DCDCDC", fontsize=FONT_BODY, wrap=30)
    draw_box(ax, outcome_x, y_gate2, outcome_w, 2.1,
             "Review needed\n(branches disagree)",
             COLOR_REVIEW, fontsize=FONT_BODY, wrap=15)

    draw_box(ax, spine_x, y_gate3, gate_w, 1.7,
             "Rule 3: fused score\n>= 0.7633 (high)?",
             "#DCDCDC", fontsize=FONT_BODY, wrap=26)
    draw_box(ax, outcome_x, y_gate3, outcome_w, 1.7,
             "High concern",
             COLOR_HIGH, fontsize=FONT_BODY, wrap=15)

    draw_box(ax, spine_x, y_gate4, gate_w, 1.7,
             "Rule 4: fused score\n<= 0.3519 (low)?",
             "#DCDCDC", fontsize=FONT_BODY, wrap=26)
    draw_box(ax, outcome_x, y_gate4, outcome_w, 1.7,
             "Limited indicators",
             COLOR_LIMITED, fontsize=FONT_BODY, wrap=15)

    draw_box(ax, spine_x, y_rule5, gate_w, 1.9,
             "Rule 5 (otherwise): score lies\nbetween the two thresholds",
             "#DCDCDC", fontsize=FONT_BODY, wrap=28)
    draw_box(ax, outcome_x, y_rule5, outcome_w, 1.9,
             "Review needed\n(between thresholds)",
             COLOR_REVIEW, fontsize=FONT_BODY, wrap=15)

    # start -> gate 1
    draw_arrow(ax, (spine_x, y_start - h_start / 2), (spine_x, y_gate1 + h_gate1 / 2))

    gate_right = spine_x + gate_w / 2
    outcome_left = outcome_x - outcome_w / 2
    label_x = (gate_right + outcome_left) / 2

    def yes_arrow(y_row):
        draw_arrow(ax, (gate_right, y_row), (outcome_left, y_row))
        label_edge(ax, label_x, y_row + 0.32, "yes", fontsize=FONT_SMALL)

    def no_arrow(y_from_box_center, half_h_from, y_to_box_center, half_h_to):
        y0 = y_from_box_center - half_h_from
        y1 = y_to_box_center + half_h_to
        draw_arrow(ax, (spine_x, y0), (spine_x, y1))
        label_edge(ax, spine_x - 0.85, (y0 + y1) / 2, "no", fontsize=FONT_SMALL)

    yes_arrow(y_gate1)
    no_arrow(y_gate1, h_gate1 / 2, y_gate2, h_gate2 / 2)

    yes_arrow(y_gate2)
    no_arrow(y_gate2, h_gate2 / 2, y_gate3, h_gate3 / 2)

    yes_arrow(y_gate3)
    no_arrow(y_gate3, h_gate3 / 2, y_gate4, h_gate4 / 2)

    yes_arrow(y_gate4)
    no_arrow(y_gate4, h_gate4 / 2, y_rule5, h_rule5 / 2)

    draw_arrow(ax, (gate_right, y_rule5), (outcome_left, y_rule5))

    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    draw_figure_3_1(out_dir / "fig3_1_architecture.png")
    draw_figure_3_2(out_dir / "fig3_2_data_and_fusion_training.png")
    draw_figure_3_3(out_dir / "fig3_3_decision_policy.png")
    print("Wrote:")
    for name in (
        "fig3_1_architecture.png",
        "fig3_2_data_and_fusion_training.png",
        "fig3_3_decision_policy.png",
    ):
        p = out_dir / name
        print(" ", p, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
