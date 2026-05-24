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

# Definizione del tipo astratto per il singolo campionamento del sensore LIDAR.
# Struttura della tupla: (angolo_campionamento, distanza_rilevata, coordinate_impatto, riferimento_ostacolo)
RaySample = Tuple[float, float, Optional[Point], Optional[Obstacle]]


@dataclass
class Environment:
    """
    Struttura dati che incapsula la topologia statica del dominio di simulazione.
    Contiene le coordinate dei nodi terminali (start/goal) e il set globale degli ostacoli geometrici.
    """
    start: Point
    goal: Point
    obstacles: List[Obstacle]


@dataclass
class SimulationArtists:
    """
    Data Transfer Object (DTO) per il raggruppamento dei reference grafici (Artists) di Matplotlib.
    Mantiene i puntatori agli oggetti del layer di presentation per consentire l'aggiornamento
    asincrono e ottimizzato (blitting/flushing) senza dover ricostruire la scena ad ogni frame.
    """
    fig: plt.Figure
    ax_map: plt.Axes
    ax_lidar_1d: plt.Axes
    stats_text: plt.Text
    lidar_line: plt.Line2D
    lidar_disc_scatter: Any
    robot_patch: patches.Circle
    sensor_border: patches.Circle
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
    """
    Buffer di stato temporale per la memorizzazione della cronologia della simulazione.
    Gestisce l'accumulo dei frame rasterizzati per l'esportazione video (GIF) e preserva
    il gradiente mnemonico (last_heuristic) necessario per la valutazione dei minimi locali.
    """
    frames: List[Image.Image] = field(default_factory=list)
    ghost_positions_x: List[float] = field(default_factory=list)
    ghost_positions_y: List[float] = field(default_factory=list)
    last_heuristic: float = float("inf")
