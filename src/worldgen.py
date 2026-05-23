from __future__ import annotations

import math
import random
from typing import List, Sequence

import config as cfg

try:
    from .geometry import Obstacle, Point
    from .sim_types import Environment
except ImportError:
    from geometry import Obstacle, Point
    from sim_types import Environment


def _build_random_non_convex_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    num_vertices = random.randint(10, 18)
    base_radius = random.uniform(cfg.MIN_OBS_SIZE, cfg.MAX_OBS_SIZE)
    margin = robot_radius + base_radius
    center_x = random.uniform(env_x_min + margin, env_x_max - margin)
    center_y = random.uniform(env_y_min + margin, env_y_max - margin)

    angle_step = 2 * math.pi / num_vertices
    points = []
    for idx in range(num_vertices):
        angle = idx * angle_step + random.uniform(-0.15, 0.15)
        radius = base_radius * random.uniform(0.9, 1.2) if idx % 2 == 0 else base_radius * random.uniform(0.3, 0.6)
        points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))

    return _build_obstacle_from_points(points, env_x_min, env_x_max, env_y_min, env_y_max, robot_radius, smooth=True)


def _build_random_convex_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    num_vertices = random.randint(12, 20)
    base_radius = random.uniform(cfg.MIN_OBS_SIZE, cfg.MAX_OBS_SIZE)
    safe_margin = robot_radius + base_radius
    center_x = random.uniform(env_x_min + safe_margin, env_x_max - safe_margin)
    center_y = random.uniform(env_y_min + safe_margin, env_y_max - safe_margin)

    raw_points = []
    for _ in range(num_vertices):
        angle = random.uniform(0, 2 * math.pi)
        radius = base_radius * random.uniform(0.6, 1.0)
        raw_points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))

    return _build_obstacle_from_points(
        _convex_hull(raw_points),
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        robot_radius,
        smooth=True,
        error_message="Inviluppo convesso fuori dai limiti.",
    )


def _build_random_non_convex_polygon_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    num_vertices = random.randint(6, 10)
    base_radius = random.uniform(cfg.MIN_OBS_SIZE, cfg.MAX_OBS_SIZE)
    safe_margin = robot_radius + base_radius
    center_x = random.uniform(env_x_min + safe_margin, env_x_max - safe_margin)
    center_y = random.uniform(env_y_min + safe_margin, env_y_max - safe_margin)

    angle_step = 2 * math.pi / num_vertices
    points = []
    for idx in range(num_vertices):
        angle = idx * angle_step + random.uniform(-0.1, 0.1)
        radius = base_radius * random.uniform(0.9, 1.1) if idx % 2 == 0 else base_radius * random.uniform(0.4, 0.6)
        points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))

    return _build_obstacle_from_points(
        points,
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        robot_radius,
        error_message="Poligono concavo fuori dai limiti.",
    )


def _build_random_rectangle_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    width = random.uniform(cfg.MIN_OBS_SIZE, cfg.MAX_OBS_SIZE)
    height = random.uniform(cfg.MIN_OBS_SIZE, cfg.MAX_OBS_SIZE)
    x_min = random.uniform(env_x_min + robot_radius, env_x_max - width - robot_radius)
    y_min = random.uniform(env_y_min + robot_radius, env_y_max - height - robot_radius)
    return Obstacle(x_min, y_min, x_min + width, y_min + height)


def _build_obstacle_from_points(
    points: Sequence[Point],
    env_x_min: float,
    env_x_max: float,
    env_y_min: float,
    env_y_max: float,
    robot_radius: float,
    *,
    smooth: bool = False,
    error_message: str = "L'ostacolo generato tocca i confini.",
) -> Obstacle:
    obstacle_points = _chaikin(list(points), iterations=2) if smooth else list(points)
    obstacle = Obstacle.from_points(obstacle_points)
    if not _obstacle_within_bounds(obstacle, env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
        raise ValueError(error_message)
    return obstacle


def _obstacle_within_bounds(
    obstacle: Obstacle,
    env_x_min: float,
    env_x_max: float,
    env_y_min: float,
    env_y_max: float,
    margin: float,
) -> bool:
    return (
        obstacle.x_min >= env_x_min + margin
        and obstacle.x_max <= env_x_max - margin
        and obstacle.y_min >= env_y_min + margin
        and obstacle.y_max <= env_y_max - margin
    )


def _chaikin(points: List[Point], iterations: int = 2) -> List[Point]:
    if len(points) < 3:
        return points
    smoothed = points[:]
    for _ in range(iterations):
        new_points = []
        for idx in range(len(smoothed)):
            p0 = smoothed[idx]
            p1 = smoothed[(idx + 1) % len(smoothed)]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_points.extend([q, r])
        smoothed = new_points
    return smoothed


def _convex_hull(points: List[Point]) -> List[Point]:
    unique_points = sorted(set(points))
    if len(unique_points) <= 2:
        return unique_points

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _generate_random_start_goal(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    margin = robot_radius * 3
    min_dist_threshold = min(env_x_max - env_x_min, env_y_max - env_y_min) * 0.65
    for _ in range(500):
        start = (
            random.uniform(env_x_min + margin, env_x_max - margin),
            random.uniform(env_y_min + margin, env_y_max - margin),
        )
        goal = (
            random.uniform(env_x_min + margin, env_x_max - margin),
            random.uniform(env_y_min + margin, env_y_max - margin),
        )
        if math.dist(start, goal) >= min_dist_threshold:
            return start, goal
    return (env_x_min + margin, env_y_min + margin), (env_x_max - margin, env_y_max - margin)


def _build_obstacle_by_kind(kind, env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    if kind == "concave":
        return _build_random_non_convex_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius)
    if kind == "convex":
        return _build_random_convex_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius)
    if kind == "polygon_concave":
        return _build_random_non_convex_polygon_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius)
    return _build_random_rectangle_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius)


def _build_world_boundaries() -> List[Obstacle]:
    wall_thickness = 2.0
    return [
        Obstacle(-wall_thickness, -wall_thickness, cfg.WIDTH + wall_thickness, 0),
        Obstacle(-wall_thickness, cfg.HEIGHT, cfg.WIDTH + wall_thickness, cfg.HEIGHT + wall_thickness),
        Obstacle(-wall_thickness, 0, 0, cfg.HEIGHT),
        Obstacle(cfg.WIDTH, 0, cfg.WIDTH + wall_thickness, cfg.HEIGHT),
    ]


def _generate_obstacles(start_position, goal_position, env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    obstacles = []
    # Usiamo un buffer fisso per la generazione, indipendente dal raggio attuale del robot


    total_obstacles = random.randint(max(6, cfg.NUM_OBSTACLES - 2), cfg.NUM_OBSTACLES + 4)

    # Zone di protezione attorno a Start e Goal (fisso)
    margin_zone = cfg.GENERATION_BUFFER * 3
    start_zone = Obstacle(
        start_position[0] - margin_zone, start_position[1] - margin_zone,
        start_position[0] + margin_zone, start_position[1] + margin_zone,
    )
    goal_zone = Obstacle(
        goal_position[0] - margin_zone, goal_position[1] - margin_zone,
        goal_position[0] + margin_zone, goal_position[1] + margin_zone,
    )

    # Calcolo tipi di ostacoli
    num_blobs = round(total_obstacles * (cfg.PCT_BLOBS / 100.0))
    num_polygons = total_obstacles - num_blobs
    num_blob_concave = round(num_blobs * (cfg.PCT_BLOB_CONCAVE / 100.0))
    num_blob_convex = num_blobs - num_blob_concave
    num_poly_concave = round(num_polygons * (cfg.PCT_POLYGON_CONCAVE / 100.0))
    num_poly_convex = num_polygons - num_poly_concave

    obstacle_kinds = (
            ["concave"] * num_blob_concave
            + ["convex"] * num_blob_convex
            + ["polygon_concave"] * num_poly_concave
            + ["polygon_convex"] * num_poly_convex
    )
    random.shuffle(obstacle_kinds)

    # Generazione ostacoli
    for kind in obstacle_kinds:
        for _ in range(200):
            try:
                # Passiamo il GENERATION_BUFFER invece di robot_radius
                new_obstacle = _build_obstacle_by_kind(
                    kind, env_x_min, env_x_max, env_y_min, env_y_max, cfg.GENERATION_BUFFER
                )
            except ValueError:
                continue

            # Verifica sovrapposizione con zone start/goal
            if new_obstacle.contains_point(start_position) or new_obstacle.contains_point(goal_position):
                continue
            if new_obstacle.intersects(start_zone, 0) or new_obstacle.intersects(goal_zone, 0):
                continue

            # Verifica collisione tra ostacoli (buffer di sicurezza fisso)
            safety_distance = (2 * (cfg.GENERATION_BUFFER + cfg.BOUNDARY_CLEARANCE) + 0.5) / 2
            if any(new_obstacle.intersects(existing, buffer=safety_distance) for existing in obstacles):
                continue

            obstacles.append(new_obstacle)
            break

    return obstacles + _build_world_boundaries()


def create_environment() -> Environment:
    env_x_min, env_x_max = 0, cfg.WIDTH
    env_y_min, env_y_max = 0, cfg.HEIGHT
    start, goal = _generate_random_start_goal(env_x_min, env_x_max, env_y_min, env_y_max, cfg.ROBOT_RADIUS)
    obstacles = _generate_obstacles(start, goal, env_x_min, env_x_max, env_y_min, env_y_max, cfg.ROBOT_RADIUS)
    return Environment(start=start, goal=goal, obstacles=obstacles)
