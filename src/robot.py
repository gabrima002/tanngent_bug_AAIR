import math
from typing import List, Optional, Tuple
import numpy as np
import config as cfg

from src.geometry import Obstacle


class Robot:
    """
    Agente autonomo basato sull'algoritmo di navigazione reattiva Tangent Bug.
    Implementa una Macchina a Stati Finiti (FSM) basata su due regimi di controllo:
    1) Motion-to-Goal: navigazione a gradiente verso il target globale o un sub-target visibile.
    2) Boundary-Following: aggiramento perimetrale attivato in presenza di minimi locali.
    """

    def __init__(
            self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float],
            max_range: Optional[float] = None, robot_radius: Optional[float] = None
    ) -> None:
        self.position = start_pos
        self.goal = goal_pos

        self.max_range = max_range if max_range is not None else cfg.SENSING_RANGE
        self.robot_radius = robot_radius if robot_radius is not None else cfg.ROBOT_RADIUS

        # Parametro di riferimento (setpoint) per il controllore di distanza dal perimetro
        self.dist_to_boundary_target = self.robot_radius + cfg.BOUNDARY_CLEARANCE

        self.heading = math.atan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0])
        self.current_behavior = "move_to_goal"

        # --- Variabili metriche proprie del Tangent Bug ---
        # d_followed: minima distanza dal goal registrata durante la circumnavigazione dell'ostacolo corrente
        self.d_followed = float("inf")
        # d_reach: minima distanza dal goal transitante per i nodi di discontinuità rilevati dal sensore
        self.d_reach = float("inf")

        # --- Registri per l'analisi dei Minimi Locali ---
        self.current_heuristic = float("inf")
        self.previous_heuristic = float("inf")
        self.best_heuristic = float("inf")

        # --- Strutture dati per il rendering analitico ---
        self.best_reach_node: Optional[Tuple[float, float]] = None
        self.best_followed_node: Optional[Tuple[float, float]] = None

        # --- Stato operativo per il Boundary Following ---
        self.followed_obstacle: Optional[Obstacle] = None
        self.follow_direction = 1  # Valore scalare per il senso di rotazione (1 = antiorario, -1 = orario)
        self.path: List[Tuple[float, float]] = [start_pos]
        self.just_switched_to_boundary_following = False

    def _choose_boundary_direction(
            self,
            closest_point: Tuple[float, float],
            obstacles: List[Obstacle],
    ) -> int:
        """
        Risoluzione euristica del senso di percorrenza ottimale per l'aggiramento.
        Campiona i due vettori tangenziali opposti alla normale della superficie e
        seleziona quello che minimizza la distanza euclidea istantanea dal target globale.
        """
        angle_to_wall = math.atan2(closest_point[1] - self.position[1], closest_point[0] - self.position[0])
        best_dir = 1
        best_score = float("inf")

        for direction in (1, -1):
            # Derivazione dell'angolo tangenziale tramite rotazione ortogonale
            tangent_angle = angle_to_wall + (math.pi / 2) * direction
            probe_step = cfg.STEP_SIZE * 0.5
            probe = (
                self.position[0] + probe_step * math.cos(tangent_angle),
                self.position[1] + probe_step * math.sin(tangent_angle),
            )

            # Reiezione della direzione qualora il vettore incida sul configuration space degli ostacoli
            if self.check_collision(probe, obstacles, extra_buffer=cfg.BOUNDARY_CLEARANCE * 0.5):
                continue

            score = math.dist(probe, self.goal)
            if score < best_score:
                best_score = score
                best_dir = direction

        return best_dir

    def _select_reach_target(self, dap, dps) -> Tuple[Optional[Tuple[float, float]], float, bool]:
        """
        Funzione di ottimizzazione locale per lo stato di Motion-to-Goal.
        Valuta i nodi di discontinuità spaziale (spigoli liberi) e restituisce
        il sub-target che minimizza la funzione di costo euristica.
        Restituisce: (Sub-target posizionale, Costo euristico, Boolean Line-of-Sight globale).
        """
        goal_visible = self.is_goal_reachable(dap)

        # Risoluzione diretta: convergenza sul target se esiste una LoS sgombra
        if goal_visible:
            return self.goal, math.dist(self.position, self.goal), True

        best_dp = None
        best_h = float("inf")

        # Culling spaziale: esclusione dei nodi eccessivamente prossimi per prevenire stalli differenziali
        min_dp_distance = max(cfg.GEOM_EPSILON, cfg.STEP_SIZE * 0.05)
        valid_dps = [
            dp for dp in dps
            if min_dp_distance < math.dist(self.position, dp) <= self.max_range
        ]

        # Funzione di costo euristica definita come: f(n) = g(n) + h(n)
        for dp in valid_dps:
            h = math.dist(self.position, dp) + math.dist(dp, self.goal)
            if h < best_h:
                best_h, best_dp = h, dp

        return best_dp, best_h, False

    def _update_move_to_goal_heuristic(self, obstacles: List[Obstacle]) -> str:
        """
        Calcola l'evoluzione della funzione di costo per l'identificazione dei minimi locali.
        Un gradiente positivo oltre la soglia configurata innesca la transizione di stato.
        """
        dap, dps = self.sense_environment(obstacles)
        best_dp, best_h, _ = self._select_reach_target(dap, dps)

        self.best_reach_node = best_dp
        self.current_heuristic = best_h

        threshold = getattr(cfg, 'HEURISTIC_THRESHOLD', 0.3)
        # Rilevamento analitico della trappola (Minimo Locale)
        if self.previous_heuristic != float("inf") and best_h > self.previous_heuristic + threshold:
            return self._switch_to_boundary(obstacles, "heuristic_increased")

        # Memorizzazione dell'optimum storico locale
        self.previous_heuristic = best_h
        if best_h < self.best_heuristic:
            self.best_heuristic = best_h

        return "move_to_goal"

    def _probe_forward_and_check_heuristic(self, obstacles: List[Obstacle]) -> str:
        """
        Procedura di svincolo per singolarità computazionali (bersaglio nullo).
        Implementa un campionamento in avanti con passo decrescente per forzare
        un aggiornamento topologico del payload sensoriale.
        """
        for step_size in (cfg.STEP_SIZE, cfg.STEP_SIZE * 0.5, cfg.STEP_SIZE * 0.25):
            probe = (
                self.position[0] + step_size * math.cos(self.heading),
                self.position[1] + step_size * math.sin(self.heading),
            )
            if self.move_robot_step(probe, obstacles, step_size=step_size):
                break

        return self._update_move_to_goal_heuristic(obstacles)

    # ----------------------------- Cinematica e Collisioni -----------------------------
    def check_collision(self, next_pos: Tuple[float, float], obstacles: List[Obstacle],
                        extra_buffer: float = 0.0) -> bool:
        """Valutazione booleana di intersezione tra la Bounding Sphere dell'agente e le primitive geometriche."""
        total_safety_radius = self.robot_radius + extra_buffer
        for obs in obstacles:
            if obs.collides_with_circle(next_pos, total_safety_radius):
                return True
        return False

    def move_robot_step(self, target_pos: Tuple[float, float], obstacles: List[Obstacle],
                        step_size: Optional[float] = None) -> bool:
        """
        Aggiornamento posizionale discreto del modello cinematico verso il target vettoriale.
        Ritorna False se la traslazione viola i vincoli spaziali (ostacoli o limiti di scena).
        """
        s = step_size if step_size is not None else cfg.STEP_SIZE
        cx, cy = self.position
        vx, vy = target_pos[0] - cx, target_pos[1] - cy
        n = math.hypot(vx, vy)

        # Risoluzione del sotto-passo terminale
        if n < s:
            nx, ny = target_pos
        else:
            nx, ny = cx + vx / n * s, cy + vy / n * s

        if self.check_collision((nx, ny), obstacles):
            return False

        # Constraint bounding-box globale per confinare l'agente
        if not (
                self.robot_radius <= nx <= cfg.WIDTH - self.robot_radius and self.robot_radius <= ny <= cfg.HEIGHT - self.robot_radius):
            return False

        if math.hypot(nx - cx, ny - cy) > 1e-9:
            self.heading = math.atan2(ny - cy, nx - cx)
            self.position = (nx, ny)
            self.path.append(self.position)
        return True

    # ----------------------------- Percezione Sensoriale -----------------------------
    def sense_environment(self, obstacles: List[Obstacle]) -> Tuple[List, List[Tuple[float, float]]]:
        """
        Simulazione del payload percettivo (LIDAR).
        Esegue un campionamento radiale discretizzato per generare il Tangent Graph locale.
        Isola e calcola l'insieme dei nodi di discontinuità spaziale (jump edges).
        """
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

        # Rilevamento analitico delle discontinuità (transizione tra ostacoli differenti o spazio vuoto)
        prev_ref = dap[-1][3] if dap else None
        for i, (_, _, pt, ref) in enumerate(dap):
            if ref != prev_ref:
                if pt: dps.append(pt)
                prev_idx = i - 1
                if dap[prev_idx][2]: dps.append(dap[prev_idx][2])
            prev_ref = ref

        # Riduzione del rumore percettivo aggregando nodi di discontinuità sub-tolleranza
        uniq = []
        for dp in dps:
            if all(math.dist(dp, u) >= cfg.DISCONTINUITY_TOL for u in uniq):
                uniq.append(dp)
        return dap, uniq

    def is_goal_reachable(self, dap) -> bool:
        """
        Verifica della Line-of-Sight (LoS) globale.
        Determina se il vettore posizionale del target giace all'interno dello spazio libero percepito.
        """
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

    # ----------------------------- Comportamenti (FSM States) -----------------------------
    def move_to_goal_behavior(self, obstacles: List[Obstacle]) -> str:
        """
        Logica di controllo per lo stato Motion-to-Goal.
        Integra una componente di sliding vettoriale (Wall-Sliding) per aggirare in modo
        fluido le asperità minime senza innescare commutazioni di stato non necessarie.
        """
        dap, dps = self.sense_environment(obstacles)
        best_dp, best_h, _ = self._select_reach_target(dap, dps)

        self.best_reach_node = best_dp
        self.current_heuristic = best_h

        if best_dp is None:
            return self._probe_forward_and_check_heuristic(obstacles)

        # Generazione del vettore di direzionalità
        dist_to_dp = math.dist(self.position, best_dp)
        angle_to_dp = math.atan2(best_dp[1] - self.position[1], best_dp[0] - self.position[0])

        # Dinamica di overstepping per il superamento angolare
        step_dp = min(cfg.STEP_SIZE, dist_to_dp)
        if dist_to_dp < cfg.STEP_SIZE * 1.5:
            step_dp = cfg.STEP_SIZE * 1.5

        next_dp_pos = (self.position[0] + step_dp * math.cos(angle_to_dp),
                       self.position[1] + step_dp * math.sin(angle_to_dp))

        # --- Risoluzione Cinematica dello Sliding Vettoriale ---
        # Se la proiezione genera impatto, il gradiente viene spurgato della sua
        # componente ortogonale alla normale della superficie colpita.
        if self.check_collision(next_dp_pos, obstacles, extra_buffer=cfg.BOUNDARY_CLEARANCE):
            min_dist_wall = float("inf")
            closest_p = None
            for obs in obstacles:
                cp = obs.get_closest_point_on_boundary(self.position)
                d = math.dist(self.position, cp)
                if d < min_dist_wall:
                    min_dist_wall, closest_p = d, cp

            if closest_p:
                # Estrazione del vettore normale
                nx, ny = self.position[0] - closest_p[0], self.position[1] - closest_p[1]
                n_len = math.hypot(nx, ny)
                if n_len > 0: nx, ny = nx / n_len, ny / n_len

                # Decomposizione vettoriale e annullamento ortogonale
                dx, dy = math.cos(angle_to_dp), math.sin(angle_to_dp)
                dot_prod = dx * nx + dy * ny
                if dot_prod < 0:
                    dx -= dot_prod * nx
                    dy -= dot_prod * ny

                # Compensazione attiva per il mantenimento del clearance
                target_dist = self.robot_radius + cfg.BOUNDARY_CLEARANCE
                if min_dist_wall < target_dist:
                    push = (target_dist - min_dist_wall) * 2.0
                    dx += nx * push
                    dy += ny * push

                # Normalizzazione del vettore risultante
                dir_len = math.hypot(dx, dy)
                if dir_len > 0: dx, dy = dx / dir_len, dy / dir_len
                next_dp_pos = (self.position[0] + dx * step_dp, self.position[1] + dy * step_dp)

        if not self.move_robot_step(next_dp_pos, obstacles):
            return self._probe_forward_and_check_heuristic(obstacles)

        if math.dist(self.position, self.goal) < cfg.GOAL_TOLERANCE:
            return "goal_reached"

        return self._update_move_to_goal_heuristic(obstacles)

    def boundary_following_behavior(self, obstacles: List[Obstacle]) -> str:
        """
        Logica di controllo per lo stato Boundary-Following.
        Implementa un controllore Proporzionale (P-Controller) accoppiato a una
        dinamica di follow-the-wall, subordinato alle Leave Conditions di transizione.
        """
        dap, dps = self.sense_environment(obstacles)

        # --- 1. Aggiornamento Visivo Globale (Calcolo d_reach) ---
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

        # --- 2. Aggiornamento Topologico Locale (Calcolo d_followed) ---
        if self.followed_obstacle:
            cp = self.followed_obstacle.get_closest_point_on_boundary(self.position)
            current_wall_dist = math.dist(cp, self.goal)
            # Acquisizione selettiva del minimo storico lungo il perimetro
            if current_wall_dist < self.d_followed:
                self.d_followed = current_wall_dist
                self.best_followed_node = cp

        # --- 3. Condizione di Uscita (Leave Condition) ---
        # Commutazione inversa se il gradiente globale percepito supera strettamente
        # il vincolo registrato nel punto di aggancio al boundary.
        if not self.just_switched_to_boundary_following and self.d_reach < (self.d_followed - cfg.LEAVE_MARGIN):
            self.previous_heuristic = float("inf")
            return "move_to_goal"

        # --- 4. Risoluzione Cinematica del Controllore P ---
        closest_pt, best_angle = None, None
        min_wall_dist = float("inf")

        # Estrazione della normale perimetrale locale
        for a_deg, dist, pt, ref in dap:
            if ref == self.followed_obstacle and pt is not None and dist < min_wall_dist:
                min_wall_dist = dist
                closest_pt = pt
                best_angle = math.radians(a_deg)

        # Fallback analitico in assenza di acquisizione sensoriale diretta
        if closest_pt is None and self.followed_obstacle is not None:
            closest_pt = self.followed_obstacle.get_closest_point_on_boundary(self.position)
            min_wall_dist = math.dist(self.position, closest_pt)
            best_angle = math.atan2(closest_pt[1] - self.position[1], closest_pt[0] - self.position[0])

        if closest_pt and best_angle is not None:
            # Segnale di errore spaziale soggetto a saturazione (clamping)
            error = min_wall_dist - self.dist_to_boundary_target
            clamped_error = max(-0.1, min(0.1, error))

            # Sintesi del gradiente tangenziale di avanzamento
            tangent_angle = best_angle + (math.pi / 2) * self.follow_direction
            tangent_dx = math.cos(tangent_angle)
            tangent_dy = math.sin(tangent_angle)

            # Sintesi del segnale di correzione normale per compensazione dell'errore
            normal_dx = math.cos(best_angle) * clamped_error
            normal_dy = math.sin(best_angle) * clamped_error

            # Combinazione lineare per la determinazione del vettore di guida
            move_dx = tangent_dx + normal_dx
            move_dy = tangent_dy + normal_dy
            move_len = math.hypot(move_dx, move_dy)

            if move_len > cfg.GEOM_EPSILON:
                move_dx /= move_len
                move_dy /= move_len

                # Esplorazione parametrica discendente del manifold spaziale
                for step_scale in (1.0, 0.5, 0.25):
                    step = cfg.STEP_SIZE * step_scale
                    nx = self.position[0] + move_dx * step
                    ny = self.position[1] + move_dy * step
                    if self.move_robot_step((nx, ny), obstacles):
                        self.just_switched_to_boundary_following = False
                        return "boundary_following"

            # Re-inizializzazione locale subordinata a stallo cinematico
            return self._switch_to_boundary(obstacles, "boundary_replan")

        return "stuck"

    def _switch_to_boundary(self, obstacles: List[Obstacle], reason: str) -> str:
        """
        Dispatcher di transizione verso lo stato di Boundary-Following.
        Gestisce il binding logico con la geometria ostruente e determina i
        parametri scalari iniziali richiesti per la circumnavigazione stabile.
        """
        entering_boundary = self.current_behavior != "boundary_following"
        if entering_boundary:
            self.just_switched_to_boundary_following = True

        min_d = float("inf")
        closest_obs = None

        # Identificazione spaziale del nodo topologico preclusivo
        for obs in obstacles:
            cp = obs.get_closest_point_on_boundary(self.position)
            d = math.dist(self.position, cp)
            if d < min_d: min_d, closest_obs = d, obs

        if closest_obs:
            previous_obstacle = self.followed_obstacle
            self.followed_obstacle = closest_obs
            cp = closest_obs.get_closest_point_on_boundary(self.position)

            # Resetting del vettore mnemonico al punto di transizione
            self.d_followed = math.dist(cp, self.goal)
            self.best_followed_node = cp
            self.best_reach_node = None

            dap, _ = self.sense_environment(obstacles)
            obs_pts = [pt for _, dist, pt, ref in dap if
                       ref == closest_obs and pt is not None and dist < self.max_range]

            # Inizializzazione dati per il rendering asincrono del proxy vettoriale
            if obs_pts:
                best_escape_pt = min(obs_pts, key=lambda p: math.dist(self.position, p) + math.dist(p, self.goal))
                self.best_reach_node = best_escape_pt

            # Assegnazione persistente della polarità direttrice subordinata alla validazione ostacolo
            if previous_obstacle is not closest_obs:
                self.follow_direction = self._choose_boundary_direction(cp, obstacles)

            return "boundary_following"

        return "stuck"

    def step(self, obstacles: List[Obstacle]) -> str:
        """
        Driver atomico della Macchina a Stati Finiti (FSM).
        Valuta i vincoli globali di terminazione e provvede al dispatching logico
        basato sul flag di stato corrente dell'architettura.
        """
        if math.dist(self.position, self.goal) < cfg.GOAL_TOLERANCE:
            return "goal_reached"

        if self.current_behavior == "move_to_goal":
            self.current_behavior = self.move_to_goal_behavior(obstacles)
        elif self.current_behavior == "boundary_following":
            self.current_behavior = self.boundary_following_behavior(obstacles)

        return self.current_behavior
