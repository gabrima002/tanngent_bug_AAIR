from __future__ import annotations

import math
from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import config as cfg

try:
    from .geometry import Obstacle, Point
    from .robot import Robot
    from .sim_types import Environment, RaySample, SimulationArtists, SnapshotState
except ImportError:
    from geometry import Obstacle, Point
    from robot import Robot
    from sim_types import Environment, RaySample, SimulationArtists, SnapshotState


def _obstacle_patch(obstacle: Obstacle) -> patches.Polygon:
    """Genera una rappresentazione poligonale (Artist) di un ostacolo per il rendering 2D."""
    return patches.Polygon(obstacle.vertices, closed=True, facecolor="lightgray", edgecolor="dimgray")


def _add_static_map_elements(ax_map: plt.Axes, environment: Environment) -> None:
    """Renderizza gli elementi statici dell'ambiente (Start point e Goal area)."""
    ax_map.add_patch(
        patches.Circle(environment.start, radius=cfg.ROBOT_RADIUS * 0.8, color="tab:blue", alpha=0.6, label="Start", zorder=10)
    )
    ax_map.add_patch(
        patches.Circle(environment.goal, radius=cfg.GOAL_TOLERANCE * 0.5, color="green", alpha=0.3, label="Goal Area", zorder=10)
    )
    ax_map.text(environment.start[0], environment.start[1] + 0.3, "START", color="tab:blue", fontsize=8, fontweight="bold", ha="center")
    ax_map.text(environment.goal[0], environment.goal[1] + 0.3, "GOAL", color="green", fontsize=8, fontweight="bold", ha="center")


def _add_legend(fig: plt.Figure, path_line, reach_line, followed_line, heuristic_line, sensor_border, disc_plot, heading_line) -> None:
    """
    Costruisce e posiziona la legenda esplicativa per la decodifica dei simboli grafici.
    Utilizza oggetti Patch per definire le chiavi di colore e oggetti Artist per le linee.
    """
    fig.legend(
        [
            Patch(facecolor="tab:blue", edgecolor="black"),
            Patch(facecolor="tab:orange", edgecolor="black"),
            path_line,
            heading_line,
            Patch(facecolor="cyan", edgecolor="black", alpha=0.5),
            disc_plot,
            reach_line,
            followed_line,
            heuristic_line,
        ],
        [
            "Stato: Move-to-goal",
            "Stato: Boundary-following",
            "Path",
            "Heading",
            "Area Sensore",
            "Discontinuità",
            "d_reach",
            "d_followed",
            "Euristica",
        ],
        loc="center left",
        bbox_to_anchor=(0.01, 0.5),
        fontsize=8,
        ncol=1,
        frameon=True,
        title="LEGENDA",
        title_fontsize=9
    )


def setup_figure(environment: Environment, robot: Robot) -> SimulationArtists:
    """
    Inizializzazione dell'interfaccia grafica.
    Configura il layout GridSpec, i subplot dedicati alla vista globale e al grafo 1D del LIDAR,
    e istanzia gli oggetti grafici (Artists) manipolabili durante la simulazione.
    """
    fig = plt.figure(figsize=(15, 8.5))

    # GridSpec: layout bi-dimensionale per separare i grafici LIDAR dalle statistiche
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0], height_ratios=[1.2, 0.25])

    # Subplot 1: Vista Globale (mappa 2D dell'ambiente)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.set_aspect("equal")
    ax_map.set_xlim(0, cfg.WIDTH)
    ax_map.set_ylim(0, cfg.HEIGHT)
    ax_map.set_title("Tangent Bug: Global 2D Environment")

    # Subplot 2: LIDAR Tangent Graph (Visuale 1D per analisi di discontinuità)
    ax_lidar_1d = fig.add_subplot(gs[0, 1])
    ax_lidar_1d.set_title("LIDAR Local Tangent Graph $L(\\theta)$")
    ax_lidar_1d.set_xlim(0, 360)
    ax_lidar_1d.set_ylim(0, cfg.SENSING_RANGE + 0.2)
    ax_lidar_1d.set_xlabel("Theta (degrees)")
    ax_lidar_1d.set_ylabel("Distance")
    ax_lidar_1d.grid(True, linestyle=":", alpha=0.6)

    lidar_line, = ax_lidar_1d.plot([], [], ".", markersize=3, color="#1f77b4", label="Muro")
    lidar_disc_scatter = ax_lidar_1d.scatter([], [], s=80, zorder=5, edgecolors="black", linewidth=1,
                                             label="Discontinuità $O_i$")
    ax_lidar_1d.legend(loc="upper right", fontsize=9)

    # Subplot 3: Pannello Statistiche dinamiche
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis("off")

    # Inserimento del testo con Bounding Box dinamico per adattamento automatico alle dimensioni della stringa
    stats_text = ax_stats.text(
        0.5, 0.5, "",
        transform=ax_stats.transAxes,
        ha="center", va="center",
        fontsize=11,
        family="monospace",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f0f0", edgecolor="black", linewidth=1.5)
    )

    # Inizializzazione entità statiche e dinamiche
    _add_static_map_elements(ax_map, environment)
    for obstacle in environment.obstacles:
        ax_map.add_patch(_obstacle_patch(obstacle))

    ghost_scatter, = ax_map.plot([], [], "o", color="purple", alpha=0.4, markersize=6, zorder=12)
    active_ghost, = ax_map.plot([], [], "o", markeredgecolor="red", markerfacecolor="none", markeredgewidth=2,
                                markersize=14, zorder=14)

    # Rendering dinamico del robot (geometria polare/cerchio)
    robot_patch = patches.Circle(robot.position, radius=0.15, facecolor="tab:blue", edgecolor="black", zorder=50)
    ax_map.add_patch(robot_patch)
    path_line, = ax_map.plot([], [], "-", linewidth=1.5, color="tab:blue", alpha=0.6, zorder=5)
    heading_line, = ax_map.plot([], [], color="gold", linewidth=2, zorder=16)
    sensor_rays = [ax_map.plot([], [], color="cyan", linewidth=0.2, alpha=0.6, zorder=4)[0] for _ in
                   range(cfg.LIDAR_NUM_RAYS)]

    # Rendering del raggio di scansione (LIDAR range)
    sensor_circle = patches.Circle(robot.position, radius=robot.robot_radius, facecolor="cyan", alpha=0.1, zorder=3)
    ax_map.add_patch(sensor_circle)

    sensor_border = patches.Circle(robot.position, radius=robot.robot_radius, edgecolor="cyan", ls="--", fill=False,
                                   lw=1.5, alpha=0.6, zorder=4)
    ax_map.add_patch(sensor_border)

    # Visualizzazione delle componenti Tangent Bug (Euristica, d_reach, d_followed)
    reach_line, = ax_map.plot([], [], ":", color="red", linewidth=1.5, alpha=0.8, zorder=21)
    followed_line, = ax_map.plot([], [], ":", color="blue", linewidth=1.5, alpha=0.8, zorder=21)
    heuristic_line, = ax_map.plot([], [], "--", color="darkorange", linewidth=2.5, alpha=1.0, zorder=30)
    disc_plot, = ax_map.plot([], [], "rx", markersize=5, zorder=20)

    # Indicatore di stato comportamentale attivo
    behavior_text = ax_map.text(0.98, 0.02, "", transform=ax_map.text(0, 0, "").get_transform() if not hasattr(ax_map,
                                                                                                               'transAxes') else ax_map.transAxes,
                                ha="right", bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"), zorder=25)

    _add_legend(fig, path_line, reach_line, followed_line, heuristic_line, sensor_border, disc_plot, heading_line)
    plt.tight_layout(rect=[0.12, 0, 1, 0.96])
    plt.ion()
    plt.show()

    return SimulationArtists(
        fig=fig,
        ax_map=ax_map,
        ax_lidar_1d=ax_lidar_1d,
        stats_text=stats_text,
        lidar_line=lidar_line,
        lidar_disc_scatter=lidar_disc_scatter,
        robot_patch=robot_patch,
        path_line=path_line,
        sensor_border=sensor_border,
        heading_line=heading_line,
        sensor_rays=sensor_rays,
        sensor_circle=sensor_circle,
        reach_line=reach_line,
        followed_line=followed_line,
        heuristic_line=heuristic_line,
        disc_plot=disc_plot,
        behavior_text=behavior_text,
        ghost_scatter=ghost_scatter,
        active_ghost=active_ghost,
    )


def update_robot_artists(artists: SimulationArtists, robot: Robot) -> None:
    """Aggiorna le coordinate e gli attributi visivi del modello del robot ad ogni step."""
    artists.robot_patch.set_center(robot.position)
    artists.robot_patch.set_facecolor("tab:blue" if robot.current_behavior == "move_to_goal" else "tab:orange")
    artists.robot_patch.set_visible(True)
    artists.robot_patch.set_zorder(100)

    artists.sensor_circle.set_radius(robot.robot_radius)
    artists.sensor_circle.set_center(robot.position)
    artists.sensor_border.set_radius(robot.robot_radius)
    artists.sensor_border.set_center(robot.position)

    if len(robot.path) > 1:
        xs, ys = zip(*robot.path)
        artists.path_line.set_data(xs, ys)
    heading_x = robot.position[0] + cfg.SENSING_RANGE * math.cos(robot.heading)
    heading_y = robot.position[1] + cfg.SENSING_RANGE * math.sin(robot.heading)
    artists.heading_line.set_data([robot.position[0], heading_x], [robot.position[1], heading_y])


def _update_sensor_rays(sensor_rays: List[plt.Line2D], old_position: Point, dap: List[RaySample]) -> None:
    """Aggiorna il raggio vettoriale di scansione LIDAR per la visualizzazione."""
    for ray_artist, (angle_deg, distance, point, _) in zip(sensor_rays, dap):
        target = point if point is not None else (
            old_position[0] + distance * math.cos(math.radians(angle_deg)),
            old_position[1] + distance * math.sin(math.radians(angle_deg)),
        )
        ray_artist.set_data([old_position[0], target[0]], [old_position[1], target[1]])


def _update_reach_visuals(artists: SimulationArtists, robot: Robot, old_position: Point) -> None:
    """Aggiorna le linee analitiche del Tangent Bug (euristiche e nodi di fuga)."""
    if robot.best_reach_node:
        artists.heuristic_line.set_data([old_position[0], robot.best_reach_node[0]], [old_position[1], robot.best_reach_node[1]])
        if robot.current_behavior == "boundary_following":
            artists.reach_line.set_data([robot.best_reach_node[0], robot.goal[0]], [robot.best_reach_node[1], robot.goal[1]])
        else:
            artists.reach_line.set_data([], [])
    else:
        artists.reach_line.set_data([], [])
        artists.heuristic_line.set_data([], [])

    if robot.current_behavior == "boundary_following" and robot.best_followed_node:
        artists.followed_line.set_data([robot.best_followed_node[0], robot.goal[0]], [robot.best_followed_node[1], robot.goal[1]])
    else:
        artists.followed_line.set_data([], [])


def _update_discontinuity_visuals(artists: SimulationArtists, old_position: Point, dap: List[RaySample], dps: List[Point]) -> None:
    """Aggiorna la visualizzazione del grafo 1D del LIDAR e la proiezione delle discontinuità."""
    if dps:
        artists.disc_plot.set_data([point[0] for point in dps], [point[1] for point in dps])
    else:
        artists.disc_plot.set_data([], [])

    plot_angles = [angle for angle, _, point, _ in dap if point is not None]
    plot_dists = [distance for _, distance, point, _ in dap if point is not None]
    artists.lidar_line.set_data(plot_angles, plot_dists)

    disc_angles = []
    disc_dists = []
    for point in dps:
        disc_dists.append(math.dist(old_position, point))
        disc_angles.append(math.degrees(math.atan2(point[1] - old_position[1], point[0] - old_position[0])) % 360)
    artists.lidar_disc_scatter.set_offsets(np.c_[disc_angles, disc_dists] if disc_angles else np.empty((0, 2)))


def update_snapshot_artists(
        artists: SimulationArtists,
        snapshot_state: SnapshotState,
        robot: Robot,
        old_position: Point,
        dap: List[RaySample],
        dps: List[Point],
) -> None:
    """Funzione gateway per l'aggiornamento sincronizzato di tutti i layer grafici ad ogni step temporale."""
    _update_sensor_rays(artists.sensor_rays, old_position, dap)
    _update_reach_visuals(artists, robot, old_position)
    _update_discontinuity_visuals(artists, old_position, dap, dps)


def _total_path_length(path: List[Point]) -> float:
    """Calcola la lunghezza totale cumulativa della traiettoria percorsa dall'agente."""
    if len(path) < 2:
        return 0.0
    return sum(math.dist(path[idx], path[idx + 1]) for idx in range(len(path) - 1))


def _format_metric(value: float) -> str:
    """Formatta i valori metrici con troncamento decimale per la visualizzazione a cruscotto."""
    return f"{value:.2f}" if value != float("inf") else "INF"


def current_heuristic_from_robot(robot: Robot, dap: List[RaySample], dps: List[Point]) -> float:
    """Interfaccia di estrazione del valore euristico corrente dal payload del robot."""
    if hasattr(robot, "_select_reach_target"):
        _, heuristic, _ = robot._select_reach_target(dap, dps)
        return heuristic
    if hasattr(robot, "_select_reach_node"):
        _, heuristic = robot._select_reach_node(dap, dps)
        return heuristic
    return getattr(robot, "current_heuristic", float("inf"))


def update_stats_text(artists: SimulationArtists, robot: Robot, step: int, current_heuristic: float,
                      previous_heuristic: float) -> float:
    """
    Gestisce l'output testuale delle metriche di simulazione.
    Esegue il formattaggio delle stringhe e la gestione del Layout dinamico tramite il Bounding Box.
    """
    total_distance = _total_path_length(robot.path)
    reach_str = _format_metric(robot.d_reach)
    follow_str = _format_metric(robot.d_followed)
    heuristic_str = _format_metric(current_heuristic)
    previous_heuristic_str = _format_metric(previous_heuristic)

    delta_heuristic = current_heuristic - previous_heuristic if all(
        math.isfinite(value) for value in (current_heuristic, previous_heuristic)) else float("inf")

    # Gestione del color encoding del delta_h
    if math.isfinite(delta_heuristic):
        delta_heuristic_str = f"{delta_heuristic:+.3f}"
    else:
        delta_heuristic_str = "N/A"

    exit_condition = ""
    if robot.current_behavior == "boundary_following" and robot.d_followed != float("inf"):
        exit_condition = f"   |   exit se: d_reach < {robot.d_followed - cfg.LEAVE_MARGIN:.2f}"

    artists.behavior_text.set_text(f"Step: {step} | State: {robot.current_behavior.upper()}")

    # Composizione del layout testuale su due righe per garantire la leggibilità
    line1 = "   |   ".join([
        f"h_curr : {heuristic_str}",
        f"h_min : {previous_heuristic_str}",
        f"delta_h : {delta_heuristic_str}"
    ])

    line2 = "   |   ".join([
        f"d_reach : {reach_str}",
        f"d_followed : {follow_str}",
        f"Distanza: {total_distance:.2f}"
    ]) + exit_condition

    artists.stats_text.set_text(f"{line1}\n{line2}")

    return total_distance
