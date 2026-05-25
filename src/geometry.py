import math
from typing import List, Optional, Sequence, Tuple
import config as cfg

# Rappresentazione vettoriale nel piano cartesiano 2D
Point = Tuple[float, float]


# =====================================================================
# OPERATORI MATEMATICI E CALCOLO VETTORIALE 2D
# =====================================================================

def _dot(a: Point, b: Point) -> float:
    """Calcola il prodotto scalare nello spazio euclideo bidimensionale."""
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: Point, b: Point) -> Point:
    """Esegue l'operazione algebrica di sottrazione tra due vettori (vettore differenza)."""
    return (a[0] - b[0], a[1] - b[1])


def _cross(a: Point, b: Point) -> float:
    """
    Calcola il determinante della matrice 2x2 formata dai due vettori.
    Equivale al modulo del prodotto vettoriale.
    """
    return a[0] * b[1] - a[1] * b[0]


def _distance_point_to_segment(point: Point, a: Point, b: Point) -> float:
    """
    Calcola la minima distanza euclidea tra un punto spaziale e un segmento definito dai vertici 'a' e 'b'.
    L'algoritmo calcola la proiezione ortogonale del punto sulla retta di supporto,
    per poi limitare (clamping) il parametro di interpolazione scalare 't' nell'intervallo [0, 1].
    """
    ab = _sub(b, a)
    ap = _sub(point, a)
    denom = _dot(ab, ab)

    # Gestione della singolarità: degenerazione del segmento in un singolo punto (lunghezza nulla)
    if denom <= cfg.GEOM_EPSILON:
        return math.dist(point, a)

    # Clamping del fattore di interpolazione parametrica 't'
    t = max(0.0, min(1.0, _dot(ap, ab) / denom))

    # Derivazione delle coordinate del punto proiettato sul segmento
    closest = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return math.dist(point, closest)


def _closest_point_on_segment(point: Point, a: Point, b: Point) -> Point:
    """
    Variante computazionale di _distance_point_to_segment che restituisce direttamente
    il vettore posizionale (coordinate cartesiane) della proiezione ortogonale sul segmento.
    """
    ab = _sub(b, a)
    ap = _sub(point, a)
    denom = _dot(ab, ab)

    # Gestione della singolarità del denominatore
    if denom <= cfg.GEOM_EPSILON:
        return a

    # Calcolo topologico tramite LERP (Linear Interpolation)
    t = max(0.0, min(1.0, _dot(ap, ab) / denom))
    return (a[0] + ab[0] * t, a[1] + ab[1] * t)


def _orientation(a: Point, b: Point, c: Point) -> float:
    """
    Valuta l'orientamento della terna ordinata di punti (a, b, c).
    Restituisce un valore scalare:
    - Positivo: orientamento antiorario (curva a sinistra).
    - Negativo: orientamento orario (curva a destra).
    - Prossimo a zero: punti collineari.
    """
    return _cross(_sub(b, a), _sub(c, a))


def _on_segment(point: Point, a: Point, b: Point) -> bool:
    """
    Verifica di appartenenza collineare: stabilisce se un punto, precedentemente
    determinato come collineare alla retta passante per 'a' e 'b', giace all'interno
    della Bounding Box locale definita dai due vertici.
    """
    eps = cfg.GEOM_EPSILON
    return (
            min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps
            and abs(_orientation(a, b, point)) <= eps
    )


def _segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """
    Rilevamento di intersezione tra due segmenti planari.
    Analizza i determinanti combinati per discriminare le intersezioni proprie
    e valuta la collinearità per la risoluzione di intersezioni improprie (sovrapposizioni).
    """
    eps = cfg.GEOM_EPSILON
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True

    if abs(o1) <= eps and _on_segment(b1, a1, a2):
        return True
    if abs(o2) <= eps and _on_segment(b2, a1, a2):
        return True
    if abs(o3) <= eps and _on_segment(a1, b1, b2):
        return True
    if abs(o4) <= eps and _on_segment(a2, b1, b2):
        return True

    return False


def _segment_distance(a1: Point, a2: Point, b1: Point, b2: Point) -> float:
    """
    Valutazione della distanza minima di separazione geometrica tra due segmenti.
    Ritorna zero qualora il test analitico rilevi un'intersezione.
    """
    if _segments_intersect(a1, a2, b1, b2):
        return 0.0

    # La distanza minima tra due segmenti non incidenti corrisponde alla minima
    # proiezione ortogonale possibile tra un vertice e il segmento opposto.
    return min(
        _distance_point_to_segment(a1, b1, b2),
        _distance_point_to_segment(a2, b1, b2),
        _distance_point_to_segment(b1, a1, a2),
        _distance_point_to_segment(b2, a1, a2),
    )


# =====================================================================
# STRUTTURE DATI SPAZIALI E TOPOLOGIA DEGLI OSTACOLI
# =====================================================================

class Obstacle:
    """
    Struttura dati per la modellazione geometrica di ostacoli planari poligonali.
    Integra algoritmi per il collision detection (AABB, Narrow-phase) e per
    la risoluzione analitica del raycasting sensoriale.
    """

    def __init__(self, x_min=None, y_min=None, x_max=None, y_max=None, points=None):
        # Istanziazione topologica da array di coordinate vettoriali
        if points is not None:
            self.vertices = [(float(x), float(y)) for x, y in points]
        # Istanziazione da primitive assi-allineate (Axis-Aligned Bounding Box)
        else:
            if None in (x_min, y_min, x_max, y_max):
                raise ValueError("L'istanziazione rettangolare richiede vincoli spaziali definiti.")
            self.vertices = [
                (float(x_min), float(y_min)), (float(x_max), float(y_min)),
                (float(x_max), float(y_max)), (float(x_min), float(y_max)),
            ]

        # Inizializzazione della Bounding Box locale per Broad-phase culling
        xs = [p[0] for p in self.vertices]
        ys = [p[1] for p in self.vertices]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)

    @classmethod
    def from_points(cls, points: Sequence[Point]) -> "Obstacle":
        """Costruttore factory polimorfico basato su array sequenziale di vertici."""
        return cls(points=points)

    def edges(self) -> List[Tuple[Point, Point]]:
        """Estrae un iteratore ciclico dei segmenti perimetrali (edge) della geometria."""
        return list(zip(self.vertices, self.vertices[1:] + [self.vertices[0]]))

    def contains_point(self, point: Point) -> bool:
        """
        Analisi di localizzazione spaziale (Point-in-Polygon).
        Implementa l'algoritmo Ray-Casting (Even-Odd Rule) derivato dal Teorema
        della Curva di Jordan per determinare l'inclusione topologica.
        """
        # Verifica di limit-case: il vettore giace sul perimetro (boundary)
        for a, b in self.edges():
            if _on_segment(point, a, b):
                return True

        inside = False
        px, py = point

        # Emissione di un raggio unidirezionale orizzontale e calcolo delle intersezioni
        for (x1, y1), (x2, y2) in self.edges():
            intersects = ((y1 > py) != (y2 > py)) and (
                    px < (x2 - x1) * (py - y1) / ((y2 - y1) + cfg.GEOM_EPSILON) + x1
            )
            if intersects:
                inside = not inside
        return inside

    def intersects(self, other_obstacle: "Obstacle", buffer: Optional[float] = None) -> bool:
        """
        Valutazione booleana di collisione inter-poligono con espansione radiale (buffer).
        Utilizza un paradigma bifase (Broad-phase AABB seguito da Narrow-phase segmentale).
        """
        b = buffer if buffer is not None else cfg.COLLISION_BUFFER

        # Broad-phase collision detection: scarta geometrie mutuamente esclusive tramite AABB dilatata
        if (self.x_max + b < other_obstacle.x_min or self.x_min - b > other_obstacle.x_max or
                self.y_max + b < other_obstacle.y_min or self.y_min - b > other_obstacle.y_max):
            return False

        # Narrow-phase collision detection: valutazione intensiva della tolleranza inter-segmentale
        for a1, a2 in self.edges():
            for b1, b2 in other_obstacle.edges():
                if _segment_distance(a1, a2, b1, b2) <= b:
                    return True

        # Test di mutua inclusione (risoluzione del caso in cui una geometria è interamente contenuta nell'altra)
        if self.contains_point(other_obstacle.vertices[0]) or other_obstacle.contains_point(self.vertices[0]):
            return True

        return False

    def intersect_ray(self, origin: Point, angle_rad: float):
        """
        Risoluazione analitica del Raycasting generato dal sensore dell'agente.
        Determina il vettore di impatto esatto e il fattore scalare di estensione 't'.
        """
        # Ottimizzazione Culling spaziale: sfrutta l'AABB per evitare la risoluzione matriciale
        # di ostacoli situati oltre la soglia operativa massima del LIDAR.
        dx_box = max(self.x_min - origin[0], 0, origin[0] - self.x_max)
        dy_box = max(self.y_min - origin[1], 0, origin[1] - self.y_max)
        if math.hypot(dx_box, dy_box) > cfg.SENSING_RANGE:
            return None, None, None

        dx, dy = math.cos(angle_rad), math.sin(angle_rad)
        ray_dir = (dx, dy)
        best_t, best_point = None, None

        # Risoluzione del sistema lineare per l'intersezione retta-segmento
        for a, b in self.edges():
            seg_dir = _sub(b, a)
            denom = _cross(ray_dir, seg_dir)

            # Scarto di vettori paralleli (determinante tendente a zero)
            if abs(denom) <= cfg.GEOM_EPSILON:
                continue

            ao = _sub(a, origin)
            t = _cross(ao, seg_dir) / denom  # Proiezione parametrica lungo il raggio di origine
            u = _cross(ao, ray_dir) / denom  # Proiezione parametrica lungo il boundary del poligono

            # Validazione spaziale: il punto di incidenza deve esistere sull'estensione del raggio (t > eps)
            # ed essere strettamente confinato tra i vertici topologici del segmento (0 <= u <= 1)
            if t > cfg.RAY_INTERSECT_EPS and -cfg.GEOM_EPSILON <= u <= 1.0 + cfg.GEOM_EPSILON:
                if best_t is None or t < best_t:
                    best_t = t
                    best_point = (origin[0] + t * dx, origin[1] + t * dy)

        # Ritorna fallimento della query se nessuna proiezione risulta geometricamente valida
        if best_t is None:
            return None, None, None

        return best_t, best_point, self

    def collides_with_circle(self, center: Point, radius: float) -> bool:
        """
        Rilevamento di collisione continua spaziale tra la topologia dell'ostacolo
        e una circonferenza di raggio noto (Rappresentazione del corpo dell'agente).
        """
        if self.contains_point(center):
            return True

        # Condizione di scarto: la minima distanza segmentale dal baricentro scende sotto la soglia radiale
        return any(_distance_point_to_segment(center, a, b) <= radius for a, b in self.edges())

    def get_closest_point_on_boundary(self, query_point: Point) -> Point:
        """
        Ricerca esaustiva della minima proiezione ortogonale disponibile sul
        boundary della geometria rispetto a un vettore posizionale esterno di query.
        Impiegato operativamente nei calcoli differenziali del Boundary Following.
        """
        best_point = self.vertices[0]
        best_dist = float("inf")

        for a, b in self.edges():
            candidate = _closest_point_on_segment(query_point, a, b)
            dist = math.dist(query_point, candidate)

            # Minimizzazione iterativa dello scalare di distanza euclidea
            if dist < best_dist:
                best_dist = dist
                best_point = candidate

        return best_point
