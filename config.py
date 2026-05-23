# =================================================================
# CONFIGURAZIONE CENTRALIZZATA TANGENT BUG
# =================================================================

# --- Ambiente e Simulazione ---
WIDTH = 20.0
HEIGHT = 20.0
NUM_OBSTACLES = 10
SEED = 67
MAX_STEPS = 2000

# --- Generazione Ostacoli  ---
MIN_OBS_SIZE = 2.0
MAX_OBS_SIZE = 5.0
PCT_BLOBS = 50
PCT_BLOB_CONCAVE = 50
PCT_POLYGON_CONCAVE = 50
GENERATION_BUFFER = 0.3

# --- Geometria e Collisioni ---
COLLISION_BUFFER = 0.2
GEOM_EPSILON = 1e-9
RAY_INTERSECT_EPS = 1e-6

# --- Sensori (LIDAR) ---
SENSING_RANGE = 1.0
LIDAR_NUM_RAYS = 90
LIDAR_FOV_DEG = 360.0
LIDAR_RESOLUTION_DEG = LIDAR_FOV_DEG / LIDAR_NUM_RAYS
DISCONTINUITY_TOL = 0.1

# --- Robot e Navigazione ---
ROBOT_RADIUS = 0.1
STEP_SIZE = 0.15
GOAL_TOLERANCE = 0.2
BOUNDARY_CLEARANCE = 0.2
HEURISTIC_THRESHOLD = 1e-4
GOAL_REACHABLE_MARGIN = 0.5
LEAVE_MARGIN = 0.1

