from __future__ import annotations

import math
import random
from typing import List, Sequence

import config as cfg


from src.geometry import Obstacle, Point
from src.sim_types import Environment



# =====================================================================
# FUNZIONI DI GENERAZIONE DELLE FORME DEGLI OSTACOLI
# =====================================================================

def _build_random_non_convex_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    """
    Genera un ostacolo concavo (non convesso) con profilo smussato.
    Crea un poligono di base alternando raggi lunghi e corti per generare concavità,
    per poi levigarne il perimetro tramite l'algoritmo di Chaikin.
    """
    num_vertices = random.randint(10, 18)
    base_radius = random.uniform(cfg.MIN_OBS_SIZE, cfg.MAX_OBS_SIZE)
    margin = robot_radius + base_radius
    center_x = random.uniform(env_x_min + margin, env_x_max - margin)
    center_y = random.uniform(env_y_min + margin, env_y_max - margin)

    angle_step = 2 * math.pi / num_vertices
    points = []
    for idx in range(num_vertices):
        # Introduzione di una perturbazione stocastica sull'angolo per irregolarità
        angle = idx * angle_step + random.uniform(-0.15, 0.15)
        # Modulazione del raggio per alternare vertici salienti e rientranti
        radius = base_radius * random.uniform(0.9, 1.2) if idx % 2 == 0 else base_radius * random.uniform(0.3, 0.6)
        points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))

    # L'opzione smooth=True delega l'arrotondamento dei vertici alla funzione di utilità
    return _build_obstacle_from_points(points, env_x_min, env_x_max, env_y_min, env_y_max, robot_radius, smooth=True)


def _build_random_convex_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    """
    Genera un ostacolo strettamente convesso con profilo smussato.
    Campiona punti casuali all'interno di un raggio base e ne estrae
    l'inviluppo convesso (Convex Hull) per garantirne la convessità geometrica.
    """
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

    # L'algoritmo di Convex Hull esclude i vertici interni prima della costruzione del poligono
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
    """
    Genera un poligono concavo preservando l'acutità dei vertici.
    La topologia generata è simile a quella stellare/frammentata, omettendo
    lo step di interpolazione (smoothing) per generare spigoli vivi.
    """
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

    # L'omissione del parametro smooth (valore predefinito False) mantiene la spigolosità originaria
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
    """Genera un ostacolo rettangolare standard (Axis-Aligned Bounding Box)."""
    width = random.uniform(cfg.MIN_OBS_SIZE, cfg.MAX_OBS_SIZE)
    height = random.uniform(cfg.MIN_OBS_SIZE, cfg.MAX_OBS_SIZE)
    x_min = random.uniform(env_x_min + robot_radius, env_x_max - width - robot_radius)
    y_min = random.uniform(env_y_min + robot_radius, env_y_max - height - robot_radius)
    return Obstacle(x_min, y_min, x_min + width, y_min + height)


# =====================================================================
# UTILITY GEOMETRICHE E MATEMATICHE
# =====================================================================

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
    """
    Costruttore e validatore del poligono. Trasforma un set di vertici in un oggetto Obstacle,
    applica l'algoritmo di levigatura se richiesto, e assicura che l'ingombro del poligono
    rientri strettamente all'interno dell'area operativa.
    """
    obstacle_points = _chaikin(list(points), iterations=2) if smooth else list(points)
    obstacle = Obstacle.from_points(obstacle_points)
    if not _obstacle_within_bounds(obstacle, env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
        # Solleva un'eccezione catturata dal ciclo di Reject Sampling per forzare un nuovo tentativo
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
    """Valuta analiticamente se l'Axis-Aligned Bounding Box (AABB) dell'ostacolo viola i limiti di scena."""
    return (
            obstacle.x_min >= env_x_min + margin
            and obstacle.x_max <= env_x_max - margin
            and obstacle.y_min >= env_y_min + margin
            and obstacle.y_max <= env_y_max - margin
    )


def _chaikin(points: List[Point], iterations: int = 2) -> List[Point]:
    """
    Algoritmo di Chaikin (Corner Cutting).
    Genera iterativamente un'approssimazione di una curva spline chiusa smussando
    ogni segmento lineare al 25% e al 75% della sua lunghezza originale.
    """
    if len(points) < 3:
        return points
    smoothed = points[:]
    for _ in range(iterations):
        new_points = []
        for idx in range(len(smoothed)):
            p0 = smoothed[idx]
            p1 = smoothed[(idx + 1) % len(smoothed)]

            # Interpolazione lineare (LERP) per il calcolo dei nuovi vertici
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_points.extend([q, r])
        smoothed = new_points
    return smoothed


def _convex_hull(points: List[Point]) -> List[Point]:
    """
    Algoritmo Monotone Chain (Andrew's algorithm) per la generazione dell'inviluppo convesso.
    Ordina i punti lessicograficamente e costruisce l'inviluppo in due passaggi (upper e lower hull).
    """
    unique_points = sorted(set(points))
    if len(unique_points) <= 2:
        return unique_points

    def cross(origin, a, b):
        # Prodotto vettoriale 2D per valutare la direzionalità (oraria vs antioraria)
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    # Costruzione della chain inferiore
    lower = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    # Costruzione della chain superiore
    upper = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    # Concatenazione delle due hull rimuovendo i vertici ridondanti alle estremità
    return lower[:-1] + upper[:-1]


# =====================================================================
# GENERAZIONE GLOBALE DELL'AMBIENTE (MAPPA, START, GOAL)
# =====================================================================

def _generate_random_start_goal(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    """
    Determina i vettori posizionali di inizio e fine dell'agente.
    Implementa un threshold di distanza euclidea minima (65% del dominio)
    per garantire la significatività e complessità del task di navigazione.
    """
    margin = robot_radius * 3
    min_dist_threshold = min(env_x_max - env_x_min, env_y_max - env_y_min) * 0.65

    # Campionamento con limite massimo di iterazioni per prevenire starvation
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

    # Posizionamento di fallback diagonale in caso di fallimento sistematico del campionamento
    return (env_x_min + margin, env_y_min + margin), (env_x_max - margin, env_y_max - margin)


def _build_obstacle_by_kind(kind, env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    """
    Design Pattern Factory per l'istanziazione polimorfica delle topologie
    degli ostacoli in base al parametro lessicale 'kind'.
    """
    if kind == "concave":
        return _build_random_non_convex_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius)
    if kind == "convex":
        return _build_random_convex_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius)
    if kind == "polygon_concave":
        return _build_random_non_convex_polygon_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius)
    return _build_random_rectangle_obstacle(env_x_min, env_x_max, env_y_min, env_y_max, robot_radius)


def _build_world_boundaries() -> List[Obstacle]:
    """
    Definisce i vincoli spaziali hard dell'ambiente generando ostacoli lineari spessi
    allineati perimetralmente alle quote 0 e max della bounding box globale.
    """
    wall_thickness = 2.0
    return [
        Obstacle(-wall_thickness, -wall_thickness, cfg.WIDTH + wall_thickness, 0),
        Obstacle(-wall_thickness, cfg.HEIGHT, cfg.WIDTH + wall_thickness, cfg.HEIGHT + wall_thickness),
        Obstacle(-wall_thickness, 0, 0, cfg.HEIGHT),
        Obstacle(cfg.WIDTH, 0, cfg.WIDTH + wall_thickness, cfg.HEIGHT),
    ]


def _generate_obstacles(start_position, goal_position, env_x_min, env_x_max, env_y_min, env_y_max, robot_radius):
    """
    Gestore della collocazione procedurale degli ostacoli nello spazio operativo.
    Sfrutta la tecnica del Reject Sampling (Campionamento con Reiezione) per risolvere
    i vincoli geometrici di non compenetrazione (overlapping) inter-ostacolo
    e le Clear Zones attorno ai punti cardine dell'agente.
    """
    obstacles = []

    # Risoluzione del numero target di ostacoli basata sulla costante globale con deviazione locale
    total_obstacles = random.randint(max(6, cfg.NUM_OBSTACLES - 2), cfg.NUM_OBSTACLES + 4)

    # Inizializzazione geometrica delle aree di esclusione fisse per lo spawn
    margin_zone = cfg.GENERATION_BUFFER * 3
    start_zone = Obstacle(
        start_position[0] - margin_zone, start_position[1] - margin_zone,
        start_position[0] + margin_zone, start_position[1] + margin_zone,
    )
    goal_zone = Obstacle(
        goal_position[0] - margin_zone, goal_position[1] - margin_zone,
        goal_position[0] + margin_zone, goal_position[1] + margin_zone,
    )

    # Ripartizione delle varianti topologiche degli ostacoli secondo i pesi prestabiliti a configurazione
    num_blobs = round(total_obstacles * (cfg.PCT_BLOBS / 100.0))
    num_polygons = total_obstacles - num_blobs
    num_blob_concave = round(num_blobs * (cfg.PCT_BLOB_CONCAVE / 100.0))
    num_blob_convex = num_blobs - num_blob_concave
    num_poly_concave = round(num_polygons * (cfg.PCT_POLYGON_CONCAVE / 100.0))
    num_poly_convex = num_polygons - num_poly_concave

    # Generazione e shuffling stocastico dell'array delle specifiche morfologiche
    obstacle_kinds = (
            ["concave"] * num_blob_concave
            + ["convex"] * num_blob_convex
            + ["polygon_concave"] * num_poly_concave
            + ["polygon_convex"] * num_poly_convex
    )
    random.shuffle(obstacle_kinds)

    # Ciclo di generazione basato su Reject Sampling per l'instaurazione delle forme valide
    for kind in obstacle_kinds:
        # Tetto iterativo locale per l'euristica di posizionamento dell'oggetto corrente
        for _ in range(200):
            try:
                new_obstacle = _build_obstacle_by_kind(
                    kind, env_x_min, env_x_max, env_y_min, env_y_max, cfg.GENERATION_BUFFER
                )
            except ValueError:
                # Trigger per violazione delle constraint geometriche di perimetro
                continue

            # Valutazione collisione esatta puntuale contro le entità dell'agente
            if new_obstacle.contains_point(start_position) or new_obstacle.contains_point(goal_position):
                continue

            # Valutazione collisione volumetrica contro i buffer operativi di partenza/arrivo
            if new_obstacle.intersects(start_zone, 0) or new_obstacle.intersects(goal_zone, 0):
                continue

            # Determinazione del limite tollerabile tra poligoni al fine di mantenere il passaggio permeabile.
            # Richiede l'intersezione AABB/GJK espansa del safety_distance per respingere lo stack.
            safety_distance = (2 * (cfg.GENERATION_BUFFER + cfg.BOUNDARY_CLEARANCE) + 0.5) / 2
            if any(new_obstacle.intersects(existing, buffer=safety_distance) for existing in obstacles):
                continue

            # Inoltro a memoria locale se tutti i vincoli dell'iper-spazio passano con successo
            obstacles.append(new_obstacle)
            break

    # Ritorna il set completo che incorpora i constraint globali statici esterni
    return obstacles + _build_world_boundaries()


def create_environment() -> Environment:
    """
    Costruttore radice dell'astrazione dell'ambiente.
    Orchestra la gerarchia generativa e restiuisce il Domain Object Environment immutabile.
    """
    env_x_min, env_x_max = 0, cfg.WIDTH
    env_y_min, env_y_max = 0, cfg.HEIGHT

    start, goal = _generate_random_start_goal(env_x_min, env_x_max, env_y_min, env_y_max, cfg.ROBOT_RADIUS)
    obstacles = _generate_obstacles(start, goal, env_x_min, env_x_max, env_y_min, env_y_max, cfg.ROBOT_RADIUS)

    return Environment(start=start, goal=goal, obstacles=obstacles)
