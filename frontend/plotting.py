"""Matplotlib rendering of the warehouse view.

Kept out of ``app.py`` so it can be unit-tested without executing the Streamlit
script. Pure function: layout + state in, a Matplotlib ``Figure`` out.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Cell = Tuple[int, int]

_PALETTE = ["#c1272d", "#0000a7", "#008176", "#eecc16", "#b3446c", "#5d3a9b"]


def draw_warehouse(
    layout: Dict,
    robots: List[Dict],
    routes: Dict[str, List[Cell]],
    obstacle: Optional[Cell] = None,
):
    width, height = layout["width"], layout["height"]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at the top
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True, color="#dddddd", linewidth=0.5)

    for (r, c) in layout.get("blocked", []):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#4b4b4b"))
    for (r, c) in layout.get("storage", []):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#e6e0c8"))
    for label, key, colour in (
        ("P", "pickups", "#7fb069"),
        ("D", "dropoffs", "#e08e45"),
        ("C", "chargers", "#5c95c7"),
    ):
        for (r, c) in layout.get(key, []):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=colour, alpha=0.8))
            ax.text(c, r, label, ha="center", va="center", fontsize=8, weight="bold")

    for i, (_robot_id, cells) in enumerate(sorted(routes.items())):
        if not cells:
            continue
        colour = _PALETTE[i % len(_PALETTE)]
        xs = [c for (_r, c) in cells]
        ys = [r for (r, _c) in cells]
        ax.plot(xs, ys, color=colour, linewidth=2, alpha=0.75, zorder=3)

    for i, robot in enumerate(sorted(robots, key=lambda x: x.get("robot_id", ""))):
        pos = robot.get("position") or {}
        r, c = pos.get("row"), pos.get("col")
        if r is None or c is None:
            continue
        colour = _PALETTE[i % len(_PALETTE)]
        offline = robot.get("status") == "offline"
        ax.scatter([c], [r], s=260, color="#999999" if offline else colour,
                   edgecolors="black", zorder=5)
        ax.text(c, r, robot.get("robot_id", "?"), ha="center", va="center",
                fontsize=7, color="white", weight="bold", zorder=6)

    if obstacle is not None:
        r, c = obstacle
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="red", linewidth=3, zorder=7))
        ax.text(c, r, "X", ha="center", va="center", color="red",
                fontsize=11, weight="bold", zorder=8)

    ax.set_title("Warehouse view")
    fig.tight_layout()
    return fig
