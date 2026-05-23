from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

try:
    from .geometry import Obstacle, Point
except ImportError:
    from geometry import Obstacle, Point

RaySample = Tuple[float, float, Optional[Point], Optional[Obstacle]]


@dataclass
class Environment:
    start: Point
    goal: Point
    obstacles: List[Obstacle]


@dataclass
class SimulationArtists:
    fig: plt.Figure
    ax_map: plt.Axes
    ax_lidar_1d: plt.Axes
    stats_text: plt.Text
    lidar_line: plt.Line2D
    lidar_disc_scatter: Any
    robot_patch: patches.Circle
    sensor_border: patches.Circle  # <--- AGGIUNGI QUESTA RIGA
    path_line: plt.Line2D
    heading_line: plt.Line2D
    sensor_rays: List[plt.Line2D]
    sensor_circle: patches.Circle
    reach_line: plt.Line2D
    followed_line: plt.Line2D
    heuristic_line: plt.Line2D
    disc_plot: plt.Line2D
    behavior_text: plt.Text
    ghost_scatter: plt.Line2D
    active_ghost: plt.Line2D


@dataclass
class SnapshotState:
    frames: List[Image.Image] = field(default_factory=list)
    ghost_positions_x: List[float] = field(default_factory=list)
    ghost_positions_y: List[float] = field(default_factory=list)
    last_heuristic: float = float("inf")
