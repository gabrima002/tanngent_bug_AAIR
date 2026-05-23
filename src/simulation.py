from __future__ import annotations

import io  # Assicurati di avere tutti gli import necessari
import os  # <--- QUESTA È LA RIGA CHE MANCA!
import random
import matplotlib.pyplot as plt
import config as cfg

# Questi sono gli import che ti mancano e che causano l'errore
from robot import Robot
from sim_playback import finalize_status, save_frame, write_gif
from sim_render import (
    current_heuristic_from_robot,
    setup_figure,
    update_robot_artists,
    update_snapshot_artists,
    update_stats_text,
)
from sim_types import SnapshotState
from worldgen import create_environment


def run_simulation() -> None:
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)
    print(f"La cartella di salvataggio è: {os.path.abspath(media_dir)}")

    # Nota: Rimosso snapshot_steps perché ora salviamo ogni frame
    realtime_pause = 0.01 # Ridotto leggermente per simulazione più fluida a video

    if cfg.SEED is not None:
        random.seed(cfg.SEED)

    environment = create_environment()
    robot = Robot(environment.start, environment.goal, cfg.SENSING_RANGE, robot_radius=cfg.ROBOT_RADIUS)
    artists = setup_figure(environment, robot)
    snapshot_state = SnapshotState()

    status = "running"
    total_distance = 0.0

    for step in range(cfg.MAX_STEPS):
        if not plt.fignum_exists(artists.fig.number):
            break

        old_position = robot.position
        dap, dps = robot.sense_environment(environment.obstacles)
        previous_heuristic = snapshot_state.last_heuristic
        current_heuristic = current_heuristic_from_robot(robot, dap, dps)
        status = robot.step(environment.obstacles)
        snapshot_state.last_heuristic = current_heuristic

        update_robot_artists(artists, robot)

        # --- MODIFICA EFFETTUATA QUI ---
        # Salviamo SEMPRE il frame ad ogni step
        update_snapshot_artists(artists, snapshot_state, robot, old_position, dap, dps)
        save_frame(artists.fig, snapshot_state.frames)
        # -------------------------------

        total_distance = update_stats_text(artists, robot, step, current_heuristic, previous_heuristic)

        artists.fig.canvas.draw_idle()
        artists.fig.canvas.flush_events()
        plt.pause(realtime_pause)

        if status in ("goal_reached", "stuck"):
            finalize_status(artists, snapshot_state, status, total_distance)
            break

    if snapshot_state.frames and status == "goal_reached":
        write_gif(media_dir, snapshot_state.frames)

    plt.ioff()
    plt.show()