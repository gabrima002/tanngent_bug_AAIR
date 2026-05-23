from __future__ import annotations

import io
import os
from datetime import datetime
from typing import List

from PIL import Image

import config as cfg

try:
    from .sim_types import SimulationArtists, SnapshotState
except ImportError:
    from sim_types import SimulationArtists, SnapshotState


def save_frame(fig, frames: List[Image.Image]) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    frames.append(Image.open(buffer).copy())
    buffer.close()


def finalize_status(artists: SimulationArtists, snapshot_state: SnapshotState, status: str, total_distance: float) -> None:
    if status == "goal_reached":
        artists.stats_text.set_text(f" GOAL RAGGIUNTO! Distanza Finale Percorsa: {total_distance:.2f} ")
        artists.stats_text.set_color("green")
    save_frame(artists.fig, snapshot_state.frames)


def write_gif(media_dir: str, frames: List[Image.Image]) -> None:
    # 1. Legge il numero della prova corrente dal file, o parte da 1
    counter_file = "prova_counter.txt"
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            i = int(f.read().strip())
    else:
        i = 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Prova_{i}_TB_Seed{cfg.SEED}_Radius{cfg.ROBOT_RADIUS}_Rays{cfg.LIDAR_NUM_RAYS}_{timestamp}.gif"
    gif_path = os.path.join(media_dir, filename)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=True, duration=300, loop=0)
