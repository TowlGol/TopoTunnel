"""
Helpers for 3D residue display styling and hover text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ResidueRenderStyle:
    color: Optional[str]
    opacity: float
    point_size: float
    selected_scale: float = 1.0


def residue_render_style(selected: bool) -> ResidueRenderStyle:
    """Return residue point sizing/opacity rules; color comes from dataset."""
    if selected:
        return ResidueRenderStyle(
            color="#D7263D",
            opacity=1.0,
            point_size=19.0,
            selected_scale=1.18,
        )

    return ResidueRenderStyle(
        color="#FF6B6B",
        opacity=0.92,
        point_size=14.0,
        selected_scale=1.0,
    )


def combination_render_style(trend: str) -> ResidueRenderStyle:
    trend = str(trend or "neutral").lower()
    if trend == "increase":
        return ResidueRenderStyle(
            color="#2AA876",
            opacity=1.0,
            point_size=22.0,
        )
    if trend == "decrease":
        return ResidueRenderStyle(
            color="#E45756",
            opacity=1.0,
            point_size=22.0,
        )
    return ResidueRenderStyle(
        color="#5F6B7A",
        opacity=1.0,
        point_size=18.0,
    )


def format_residue_hover_text(residue_id: int, info: Optional[dict]) -> str:
    """Build tooltip text for a hovered residue."""
    if not info:
        return f"Residue #{int(residue_id)}"

    residue_label = info.get("label") or str(int(residue_id))
    path_count = int(info.get("path_count", 0) or 0)
    x = info.get("x")
    y = info.get("y")
    z = info.get("z")
    if all(value is not None and np.isfinite(value) for value in (x, y, z)):
        position_text = f"({float(x):.2f}, {float(y):.2f}, {float(z):.2f})"
    else:
        position_text = "-"

    return (
        f"Residue #{residue_label}\n"
        f"Path Count: {path_count}\n"
        f"Position: {position_text}"
    )


def format_combination_hover_text(info: Optional[dict]) -> str:
    if not info:
        return "Residue Combination"

    label = str(info.get("combination_label") or "Residue Combination")
    property_label = str(info.get("property_label") or "Property")
    property_value = info.get("property_value")
    trend = str(info.get("property_trend_label") or "")
    centroid = info.get("centroid")
    affected_path_count = int(info.get("affected_path_count", 0) or 0)

    if centroid and len(centroid) == 3 and all(np.isfinite(v) for v in centroid):
        centroid_text = f"({float(centroid[0]):.2f}, {float(centroid[1]):.2f}, {float(centroid[2]):.2f})"
    else:
        centroid_text = "-"

    property_text = "-"
    if property_value is not None and np.isfinite(float(property_value)):
        property_text = f"{float(property_value):+.3f}" if trend else f"{float(property_value):.3f}"

    lines = [label, f"Affected Paths: {affected_path_count}", f"{property_label}: {property_text}"]
    if trend:
        lines.append(f"Trend: {trend}")
    lines.append(f"Centroid: {centroid_text}")
    return "\n".join(lines)
