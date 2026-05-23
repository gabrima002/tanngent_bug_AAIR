import math
from typing import List, Optional, Sequence, Tuple
import config as cfg

Point = Tuple[float, float]

def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]

def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])

def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]

def _distance_point_to_segment(point: Point, a: Point, b: Point) -> float:
    ab = _sub(b, a)
    ap = _sub(point, a)
    denom = _dot(ab, ab)
    if denom <= cfg.GEOM_EPSILON:
        return math.dist(point, a)
    t = max(0.0, min(1.0, _dot(ap, ab) / denom))
    closest = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return math.dist(point, closest)

def _closest_point_on_segment(point: Point, a: Point, b: Point) -> Point:
    ab = _sub(b, a)
    ap = _sub(point, a)
    denom = _dot(ab, ab)
    if denom <= cfg.GEOM_EPSILON:
        return a
    t = max(0.0, min(1.0, _dot(ap, ab) / denom))
    return (a[0] + ab[0] * t, a[1] + ab[1] * t)

def _orientation(a: Point, b: Point, c: Point) -> float:
    return _cross(_sub(b, a), _sub(c, a))

def _on_segment(point: Point, a: Point, b: Point) -> bool:
    eps = cfg.GEOM_EPSILON
    return (
            min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps
            and abs(_orientation(a, b, point)) <= eps
    )

def _segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
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
    if _segments_intersect(a1, a2, b1, b2):
        return 0.0
    return min(
        _distance_point_to_segment(a1, b1, b2),
        _distance_point_to_segment(a2, b1, b2),
        _distance_point_to_segment(b1, a1, a2),
        _distance_point_to_segment(b2, a1, a2),
    )

class Obstacle:
    def __init__(self, x_min=None, y_min=None, x_max=None, y_max=None, points=None):
        if points is not None:
            self.vertices = [(float(x), float(y)) for x, y in points]
        else:
            if None in (x_min, y_min, x_max, y_max):
                raise ValueError("Rectangle obstacles require boundaries.")
            self.vertices = [
                (float(x_min), float(y_min)), (float(x_max), float(y_min)),
                (float(x_max), float(y_max)), (float(x_min), float(y_max)),
            ]

        xs = [p[0] for p in self.vertices]
        ys = [p[1] for p in self.vertices]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)

    @classmethod
    def from_points(cls, points: Sequence[Point]) -> "Obstacle":
        return cls(points=points)

    def edges(self) -> List[Tuple[Point, Point]]:
        return list(zip(self.vertices, self.vertices[1:] + [self.vertices[0]]))

    def contains_point(self, point: Point) -> bool:
        for a, b in self.edges():
            if _on_segment(point, a, b):
                return True
        inside = False
        px, py = point
        for (x1, y1), (x2, y2) in self.edges():
            intersects = ((y1 > py) != (y2 > py)) and (
                    px < (x2 - x1) * (py - y1) / ((y2 - y1) + cfg.GEOM_EPSILON) + x1
            )
            if intersects:
                inside = not inside
        return inside

    def intersects(self, other_obstacle: "Obstacle", buffer: Optional[float] = None) -> bool:
        b = buffer if buffer is not None else cfg.COLLISION_BUFFER
        if (self.x_max + b < other_obstacle.x_min or self.x_min - b > other_obstacle.x_max or
                self.y_max + b < other_obstacle.y_min or self.y_min - b > other_obstacle.y_max):
            return False
        for a1, a2 in self.edges():
            for b1, b2 in other_obstacle.edges():
                if _segment_distance(a1, a2, b1, b2) <= b:
                    return True
        if self.contains_point(other_obstacle.vertices[0]) or other_obstacle.contains_point(self.vertices[0]):
            return True
        return False

    def intersect_ray(self, origin: Point, angle_rad: float):
        # OTTIMIZZAZIONE AABB: Salta l'ostacolo se è fuori portata del sensore
        dx_box = max(self.x_min - origin[0], 0, origin[0] - self.x_max)
        dy_box = max(self.y_min - origin[1], 0, origin[1] - self.y_max)
        if math.hypot(dx_box, dy_box) > cfg.SENSING_RANGE:
            return None, None, None

        dx, dy = math.cos(angle_rad), math.sin(angle_rad)
        ray_dir = (dx, dy)
        best_t, best_point = None, None

        for a, b in self.edges():
            seg_dir = _sub(b, a)
            denom = _cross(ray_dir, seg_dir)
            if abs(denom) <= cfg.GEOM_EPSILON:
                continue

            ao = _sub(a, origin)
            t = _cross(ao, seg_dir) / denom
            u = _cross(ao, ray_dir) / denom

            if t > cfg.RAY_INTERSECT_EPS and -cfg.GEOM_EPSILON <= u <= 1.0 + cfg.GEOM_EPSILON:
                if best_t is None or t < best_t:
                    best_t = t
                    best_point = (origin[0] + t * dx, origin[1] + t * dy)

        if best_t is None:
            return None, None, None
        return best_t, best_point, self

    def collides_with_circle(self, center: Point, radius: float) -> bool:
        if self.contains_point(center):
            return True
        return any(_distance_point_to_segment(center, a, b) <= radius for a, b in self.edges())

    def get_closest_point_on_boundary(self, query_point: Point) -> Point:
        best_point = self.vertices[0]
        best_dist = float("inf")
        for a, b in self.edges():
            candidate = _closest_point_on_segment(query_point, a, b)
            dist = math.dist(query_point, candidate)
            if dist < best_dist:
                best_dist = dist
                best_point = candidate
        return best_point
