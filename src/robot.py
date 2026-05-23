import math
from typing import List, Optional, Tuple
import numpy as np
import config as cfg

try:
    from .geometry import Obstacle
except ImportError:
    from geometry import Obstacle


class Robot:
    """
    Robot con comportamento Tangent Bug Semplificato.
    """

    def __init__(
            self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float],
            max_range: Optional[float] = None, robot_radius: Optional[float] = None
    ) -> None:
        self.position = start_pos
        self.goal = goal_pos

        self.max_range = max_range if max_range is not None else cfg.SENSING_RANGE
        self.robot_radius = robot_radius if robot_radius is not None else cfg.ROBOT_RADIUS
        self.dist_to_boundary_target = self.robot_radius + cfg.BOUNDARY_CLEARANCE

        self.heading = math.atan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0])
        self.current_behavior = "move_to_goal"

        # Variabili Tangent Bug
        self.d_followed = float("inf")
        self.d_reach = float("inf")

        # Variabili Euristica Semplificata
        self.current_heuristic = float("inf")
        self.previous_heuristic = float("inf")
        self.best_heuristic = float("inf")

        # Variabili Grafica
        self.best_reach_node: Optional[Tuple[float, float]] = None
        self.best_followed_node: Optional[Tuple[float, float]] = None

        self.followed_obstacle: Optional[Obstacle] = None
        self.follow_direction = 1
        self.path: List[Tuple[float, float]] = [start_pos]
        self.just_switched_to_boundary_following = False

    def _choose_boundary_direction(
            self,
            closest_point: Tuple[float, float],
            obstacles: List[Obstacle],
    ) -> int:
        angle_to_wall = math.atan2(closest_point[1] - self.position[1], closest_point[0] - self.position[0])
        best_dir = 1
        best_score = float("inf")

        for direction in (1, -1):
            tangent_angle = angle_to_wall + (math.pi / 2) * direction
            probe_step = cfg.STEP_SIZE * 0.5
            probe = (
                self.position[0] + probe_step * math.cos(tangent_angle),
                self.position[1] + probe_step * math.sin(tangent_angle),
            )

            if self.check_collision(probe, obstacles, extra_buffer=cfg.BOUNDARY_CLEARANCE * 0.5):
                continue

            score = math.dist(probe, self.goal)
            if score < best_score:
                best_score = score
                best_dir = direction

        return best_dir

    def _select_reach_target(self, dap, dps) -> Tuple[Optional[Tuple[float, float]], float, bool]:
        goal_visible = self.is_goal_reachable(dap)

        if goal_visible:
            return self.goal, math.dist(self.position, self.goal), True

        best_dp = None
        best_h = float("inf")
        # Non scartare le discontinuita' vicine: agli spigoli sono spesso il
        # candidato corretto. Ignoriamo solo punti praticamente coincidenti col robot.
        min_dp_distance = max(cfg.GEOM_EPSILON, cfg.STEP_SIZE * 0.05)
        valid_dps = [
            dp for dp in dps
            if min_dp_distance < math.dist(self.position, dp) <= self.max_range
        ]

        for dp in valid_dps:
            h = math.dist(self.position, dp) + math.dist(dp, self.goal)
            if h < best_h:
                best_h, best_dp = h, dp

        return best_dp, best_h, False

    def _update_move_to_goal_heuristic(self, obstacles: List[Obstacle]) -> str:
        dap, dps = self.sense_environment(obstacles)
        best_dp, best_h, _ = self._select_reach_target(dap, dps)

        self.best_reach_node = best_dp
        self.current_heuristic = best_h

        threshold = getattr(cfg, 'HEURISTIC_THRESHOLD', 0.3)
        if self.previous_heuristic != float("inf") and best_h > self.previous_heuristic + threshold:
            return self._switch_to_boundary(obstacles, "heuristic_increased")

        self.previous_heuristic = best_h
        if best_h < self.best_heuristic:
            self.best_heuristic = best_h

        return "move_to_goal"

    def _probe_forward_and_check_heuristic(self, obstacles: List[Obstacle]) -> str:
        for step_size in (cfg.STEP_SIZE, cfg.STEP_SIZE * 0.5, cfg.STEP_SIZE * 0.25):
            probe = (
                self.position[0] + step_size * math.cos(self.heading),
                self.position[1] + step_size * math.sin(self.heading),
            )
            if self.move_robot_step(probe, obstacles, step_size=step_size):
                break

        return self._update_move_to_goal_heuristic(obstacles)

    # ----------------------------- Logica di Movimento Base -----------------------------
    def check_collision(self, next_pos: Tuple[float, float], obstacles: List[Obstacle],
                        extra_buffer: float = 0.0) -> bool:
        total_safety_radius = self.robot_radius + extra_buffer
        for obs in obstacles:
            if obs.collides_with_circle(next_pos, total_safety_radius):
                return True
        return False

    def move_robot_step(self, target_pos: Tuple[float, float], obstacles: List[Obstacle],
                        step_size: Optional[float] = None) -> bool:
        s = step_size if step_size is not None else cfg.STEP_SIZE
        cx, cy = self.position
        vx, vy = target_pos[0] - cx, target_pos[1] - cy
        n = math.hypot(vx, vy)

        if n < s:
            nx, ny = target_pos
        else:
            nx, ny = cx + vx / n * s, cy + vy / n * s

        if self.check_collision((nx, ny), obstacles):
            return False

        if not (
                self.robot_radius <= nx <= cfg.WIDTH - self.robot_radius and self.robot_radius <= ny <= cfg.HEIGHT - self.robot_radius):
            return False

        if math.hypot(nx - cx, ny - cy) > 1e-9:
            self.heading = math.atan2(ny - cy, nx - cx)
            self.position = (nx, ny)
            self.path.append(self.position)
        return True

    # ----------------------------- Percezione -----------------------------
    def sense_environment(self, obstacles: List[Obstacle]) -> Tuple[List, List[Tuple[float, float]]]:
        dap, dps = [], []
        res = cfg.LIDAR_RESOLUTION_DEG
        angles = np.linspace(0, 2 * math.pi, int(360 / res), endpoint=False)

        for ang in angles:
            best = self.max_range
            best_pt, best_ref = None, None
            for obs in obstacles:
                d, p, r = obs.intersect_ray(self.position, ang)
                if d is not None and d < best:
                    best, best_pt, best_ref = d, p, r
            dap.append((math.degrees(ang), best, best_pt, best_ref))

        prev_ref = dap[-1][3] if dap else None
        for i, (_, _, pt, ref) in enumerate(dap):
            if ref != prev_ref:
                if pt: dps.append(pt)
                prev_idx = i - 1
                if dap[prev_idx][2]: dps.append(dap[prev_idx][2])
            prev_ref = ref

        uniq = []
        for dp in dps:
            if all(math.dist(dp, u) >= cfg.DISCONTINUITY_TOL for u in uniq):
                uniq.append(dp)
        return dap, uniq

    def is_goal_reachable(self, dap) -> bool:
        d_goal = math.dist(self.position, self.goal)
        if d_goal < self.robot_radius * 1.1: return True
        ang_to_goal = math.degrees(math.atan2(self.goal[1] - self.position[1], self.goal[0] - self.position[0])) % 360
        min_diff, closest_dist, closest_pt = 360, self.max_range, None

        for a_deg, dist, pt, _ in dap:
            diff = abs(a_deg - ang_to_goal)
            diff = diff if diff <= 180 else 360 - diff
            if diff < min_diff:
                min_diff, closest_dist, closest_pt = diff, dist, pt

        return not (closest_pt is not None and closest_dist < d_goal - cfg.GOAL_REACHABLE_MARGIN)

    # ----------------------------- Comportamento MOVE_TO_GOAL -----------------------------
    def move_to_goal_behavior(self, obstacles: List[Obstacle]) -> str:
        dap, dps = self.sense_environment(obstacles)
        best_dp, best_h, _ = self._select_reach_target(dap, dps)

        # Aggiornamento UI
        self.best_reach_node = best_dp
        self.current_heuristic = best_h

        # Sblocco Cieco di emergenza (se il bersaglio non c'è)
        if best_dp is None:
            return self._probe_forward_and_check_heuristic(obstacles)

        # 3. MOVIMENTO E SCIVOLAMENTO (Sliding)
        dist_to_dp = math.dist(self.position, best_dp)
        angle_to_dp = math.atan2(best_dp[1] - self.position[1], best_dp[0] - self.position[0])

        # Facciamo un passo più lungo in prossimità dello spigolo per superarlo meglio
        step_dp = min(cfg.STEP_SIZE, dist_to_dp)
        if dist_to_dp < cfg.STEP_SIZE * 1.5:
            step_dp = cfg.STEP_SIZE * 1.5

        next_dp_pos = (self.position[0] + step_dp * math.cos(angle_to_dp),
                       self.position[1] + step_dp * math.sin(angle_to_dp))

        # Se andiamo addosso al buffer, calcoliamo il vettore di scivolamento
        if self.check_collision(next_dp_pos, obstacles, extra_buffer=cfg.BOUNDARY_CLEARANCE):
            min_dist_wall = float("inf")
            closest_p = None
            for obs in obstacles:
                cp = obs.get_closest_point_on_boundary(self.position)
                d = math.dist(self.position, cp)
                if d < min_dist_wall:
                    min_dist_wall, closest_p = d, cp

            if closest_p:
                nx, ny = self.position[0] - closest_p[0], self.position[1] - closest_p[1]
                n_len = math.hypot(nx, ny)
                if n_len > 0: nx, ny = nx / n_len, ny / n_len

                dx, dy = math.cos(angle_to_dp), math.sin(angle_to_dp)
                dot_prod = dx * nx + dy * ny
                if dot_prod < 0:
                    dx -= dot_prod * nx
                    dy -= dot_prod * ny

                target_dist = self.robot_radius + cfg.BOUNDARY_CLEARANCE
                if min_dist_wall < target_dist:
                    push = (target_dist - min_dist_wall) * 2.0
                    dx += nx * push
                    dy += ny * push

                dir_len = math.hypot(dx, dy)
                if dir_len > 0: dx, dy = dx / dir_len, dy / dir_len
                next_dp_pos = (self.position[0] + dx * step_dp, self.position[1] + dy * step_dp)

        # Eseguiamo il passo
        if not self.move_robot_step(next_dp_pos, obstacles):
            return self._probe_forward_and_check_heuristic(obstacles)

        # 4. SALVATAGGIO STATO E MEMORIA
        if math.dist(self.position, self.goal) < cfg.GOAL_TOLERANCE:
            return "goal_reached"

        return self._update_move_to_goal_heuristic(obstacles)

    # ----------------------------- Comportamento BOUNDARY_FOLLOWING -----------------------------
    def boundary_following_behavior(self, obstacles: List[Obstacle]) -> str:
        dap, dps = self.sense_environment(obstacles)

        # 1. Vista (d_reach)
        if self.is_goal_reachable(dap):
            self.d_reach = math.dist(self.position, self.goal)
            self.best_reach_node = self.position
        else:
            valid_dps = [dp for dp in dps if math.dist(self.position, dp) <= self.max_range]
            if valid_dps:
                best_dp = min(valid_dps, key=lambda dp: math.dist(dp, self.goal))
                self.d_reach = math.dist(best_dp, self.goal)
                self.best_reach_node = best_dp
            else:
                self.d_reach = float("inf")
                self.best_reach_node = None

        # 2. Memoria (d_followed)
        if self.followed_obstacle:
            cp = self.followed_obstacle.get_closest_point_on_boundary(self.position)
            current_wall_dist = math.dist(cp, self.goal)
            if current_wall_dist < self.d_followed:
                self.d_followed = current_wall_dist
                self.best_followed_node = cp

        # 3. CONDIZIONE DI USCITA
        if not self.just_switched_to_boundary_following and self.d_reach < (self.d_followed - cfg.LEAVE_MARGIN):
            self.previous_heuristic = float("inf")
            return "move_to_goal"

        # 4. Inseguimento Muro: usa il punto piu' vicino del bordo per stimare
        # la normale locale e poi avanza lungo la tangente. Questo evita di
        # scegliere raggi "dietro" al robot solo per mantenere la distanza.
        closest_pt, best_angle = None, None
        min_wall_dist = float("inf")
        for a_deg, dist, pt, ref in dap:
            if ref == self.followed_obstacle and pt is not None and dist < min_wall_dist:
                min_wall_dist = dist
                closest_pt = pt
                best_angle = math.radians(a_deg)

        if closest_pt is None and self.followed_obstacle is not None:
            closest_pt = self.followed_obstacle.get_closest_point_on_boundary(self.position)
            min_wall_dist = math.dist(self.position, closest_pt)
            best_angle = math.atan2(closest_pt[1] - self.position[1], closest_pt[0] - self.position[0])

        if closest_pt and best_angle is not None:
            error = min_wall_dist - self.dist_to_boundary_target
            clamped_error = max(-0.1, min(0.1, error))

            tangent_angle = best_angle + (math.pi / 2) * self.follow_direction
            tangent_dx = math.cos(tangent_angle)
            tangent_dy = math.sin(tangent_angle)

            normal_dx = math.cos(best_angle) * clamped_error
            normal_dy = math.sin(best_angle) * clamped_error

            move_dx = tangent_dx + normal_dx
            move_dy = tangent_dy + normal_dy
            move_len = math.hypot(move_dx, move_dy)

            if move_len > cfg.GEOM_EPSILON:
                move_dx /= move_len
                move_dy /= move_len

                for step_scale in (1.0, 0.5, 0.25):
                    step = cfg.STEP_SIZE * step_scale
                    nx = self.position[0] + move_dx * step
                    ny = self.position[1] + move_dy * step
                    if self.move_robot_step((nx, ny), obstacles):
                        self.just_switched_to_boundary_following = False
                        return "boundary_following"

            # Manteniamo stabile il verso scelto. Se il passo locale fallisce,
            # ricalcoliamo l'aggancio al contorno invece di ribaltare subito lato.
            return self._switch_to_boundary(obstacles, "boundary_replan")

        return "stuck"

    def _switch_to_boundary(self, obstacles: List[Obstacle], reason: str) -> str:
        entering_boundary = self.current_behavior != "boundary_following"
        if entering_boundary:
            self.just_switched_to_boundary_following = True
        min_d = float("inf")
        closest_obs = None

        for obs in obstacles:
            cp = obs.get_closest_point_on_boundary(self.position)
            d = math.dist(self.position, cp)
            if d < min_d: min_d, closest_obs = d, obs

        if closest_obs:
            previous_obstacle = self.followed_obstacle
            self.followed_obstacle = closest_obs
            cp = closest_obs.get_closest_point_on_boundary(self.position)
            self.d_followed = math.dist(cp, self.goal)
            self.best_followed_node = cp
            self.best_reach_node = None

            dap, _ = self.sense_environment(obstacles)
            obs_pts = [pt for _, dist, pt, ref in dap if
                       ref == closest_obs and pt is not None and dist < self.max_range]

            if obs_pts:
                best_escape_pt = min(obs_pts, key=lambda p: math.dist(self.position, p) + math.dist(p, self.goal))
                self.best_reach_node = best_escape_pt

            # Tangent Bug classico: il verso si sceglie una volta per ostacolo
            # e si mantiene. Se il robot rientra in boundary following sullo
            # stesso ostacolo, riusa il verso gia' scelto.
            if previous_obstacle is not closest_obs:
                self.follow_direction = self._choose_boundary_direction(cp, obstacles)

            return "boundary_following"

        return "stuck"

    def step(self, obstacles: List[Obstacle]) -> str:
        if math.dist(self.position, self.goal) < cfg.GOAL_TOLERANCE:
            return "goal_reached"

        if self.current_behavior == "move_to_goal":
            self.current_behavior = self.move_to_goal_behavior(obstacles)
        elif self.current_behavior == "boundary_following":
            self.current_behavior = self.boundary_following_behavior(obstacles)

        return self.current_behavior
