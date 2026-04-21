import json
import numpy as np
from scipy.spatial import cKDTree

TRAIN = 0
TEST = 1
NONE = -1

def get_points(file_path):
    with open(file_path, 'r') as f:
        transforms = json.load(f)
    image_frames = []
    xs, ys = [], []
    for frame in  transforms["frames"]:
        image_frames.append(frame["file_path"].split('/')[-1])
        
        transform = np.array(frame["transform_matrix"])
        R = transform[:3, :3]
        t = transform[:3, 3]
        camera_position_world = -R.T @ t
        x, y = camera_position_world[0], camera_position_world[1]
        xs.append(x)
        ys.append(y)
    return image_frames, np.stack([xs, ys], axis=1)

# def plot_split(points, labels, save_path):
#     plt.scatter(points[np.where(labels == TRAIN)][:, 0], points[np.where(labels == TRAIN)][:, 1], c='blue', label='train')
#     plt.scatter(points[np.where(labels == TEST)][:, 0], points[np.where(labels == TEST)][:, 1], c='orange', label='test')
#     plt.scatter(points[np.where(labels == NONE)][:, 0], points[np.where(labels == NONE)][:, 1], c='black', label='none')
#     plt.axis('equal')
#     plt.legend()
#     plt.savefig(save_path)


def farthest_point_sampling(points, k, start_idx=None):
    """
    Greedy farthest point sampling on 2D points.

    Parameters
    ----------
    points : (N, 2) array
    k : int
        Number of seeds to select.
    start_idx : int or None
        Optional first seed. If None, uses 0.

    Returns
    -------
    selected : list[int]
        Indices of selected FPS seeds.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0 or k <= 0:
        return []

    k = min(k, n)
    if start_idx is None:
        start_idx = 0

    selected = [start_idx]
    min_dists = np.linalg.norm(points - points[start_idx], axis=1)

    for _ in range(1, k):
        nxt = int(np.argmax(min_dists))
        selected.append(nxt)

        d = np.linalg.norm(points - points[nxt], axis=1)
        min_dists = np.minimum(min_dists, d)

    return selected


def circular_window_indices(n, center, half_width):
    """
    Return indices in a circular window around center:
    [center-half_width, ..., center+half_width] modulo n.
    """
    offs = np.arange(-half_width, half_width + 1)
    return (center + offs) % n


def mask_to_circular_runs(mask):
    """
    Convert boolean mask over a circular trajectory into runs.

    Returns list of (start, end) inclusive runs.
    If a run wraps around the end, it is returned as (start, end)
    with start > end to indicate circular wrap.
    """
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if n == 0 or not np.any(mask):
        return []

    runs = []
    in_run = False
    start = None

    for i in range(n):
        if mask[i] and not in_run:
            start = i
            in_run = True
        elif not mask[i] and in_run:
            runs.append((start, i - 1))
            in_run = False

    if in_run:
        runs.append((start, n - 1))

    # Merge first/last runs if circularly connected
    if len(runs) >= 2 and mask[0] and mask[-1]:
        first = runs[0]
        last = runs[-1]
        merged = (last[0], first[1])  # wrapped run
        runs = [merged] + runs[1:-1]

    return runs


def score_split(labels, alpha_none=0.3):
    """
    Higher is better.
    Encourages more TEST and penalizes too much NONE.
    """
    n_test = np.sum(labels == TEST)
    n_none = np.sum(labels == NONE)
    return float(n_test - alpha_none * n_none)


def build_split_from_seeds(
    points,
    seed_indices,
    min_dist=1.0,
    test_window_size=50,
):
    """
    Build train/test/none labels from given seed indices.

    Parameters
    ----------
    points : (N, 2) array
    seed_indices : list[int]
        Centers of test windows.
    min_dist : float
        Minimum spatial distance between any TEST point and any TRAIN point.
    test_window_size : int
        Number of trajectory points in each test sequence window.

    Returns
    -------
    labels : (N,) int array in {TRAIN, TEST, NONE}
    info : dict
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    labels = np.full(n, TRAIN, dtype=np.int8)

    if n == 0:
        return labels, {
            "seed_indices": [],
            "test_runs": [],
            "train_runs": [],
            "none_runs": [],
        }

    half_width = max(0, (int(test_window_size) - 1) // 2)

    # 1) Mark trajectory windows around seeds as test
    test_mask = np.zeros(n, dtype=bool)
    for s in seed_indices:
        idx = circular_window_indices(n, s % n, half_width)
        test_mask[idx] = True

    labels[test_mask] = TEST

    # 2) Spatial exclusion: points near any TEST point cannot be TRAIN
    tree = cKDTree(points)
    test_idx = np.where(test_mask)[0]

    near_test = np.zeros(n, dtype=bool)
    if len(test_idx) > 0:
        for i in test_idx:
            neighbors = tree.query_ball_point(points[i], r=min_dist)
            near_test[neighbors] = True

    labels[(near_test) & (~test_mask)] = NONE

    info = {
        "seed_indices": list(map(int, seed_indices)),
        "test_runs": mask_to_circular_runs(labels == TEST),
        "train_runs": mask_to_circular_runs(labels == TRAIN),
        "none_runs": mask_to_circular_runs(labels == NONE),
        "n_train": int(np.sum(labels == TRAIN)),
        "n_test": int(np.sum(labels == TEST)),
        "n_none": int(np.sum(labels == NONE)),
    }
    return labels, info


def spatial_sequence_split(
    points,
    min_dist=1.0,
    n_test_sequences=1,
    test_window_size=50,
    n_trials=30,
    alpha_none=0.3,
    start_idx_mode="random",
    random_state=None,
):
    """
    Split cyclic trajectory into TRAIN / TEST / NONE using:
      - farthest point sampling for seed selection
      - contiguous trajectory windows for TEST
      - spatial exclusion radius for TRAIN safety

    Parameters
    ----------
    points : (N, 2) array
        Trajectory XY points.
    min_dist : float
        Minimum allowed distance between TEST and TRAIN points.
    n_test_sequences : int
        Number of contiguous test sequences.
    test_window_size : int
        Length of each test sequence in trajectory points.
    n_trials : int
        Number of FPS initializations to try.
    alpha_none : float
        Penalty weight for NONE points in split scoring.
    start_idx_mode : {"random", "deterministic"}
        FPS needs a first point. We try multiple starts.
    random_state : int or None

    Returns
    -------
    labels : (N,) int array
        Values in {TRAIN=0, TEST=1, NONE=-1}
    best_info : dict
        Extra info, including seed indices and contiguous runs.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    n = len(points)
    if n == 0:
        return np.array([], dtype=np.int8), {
            "seed_indices": [],
            "test_runs": [],
            "train_runs": [],
            "none_runs": [],
            "score": 0.0,
            "n_train": 0,
            "n_test": 0,
            "n_none": 0,
        }

    rng = np.random.default_rng(random_state)
    n_trials = max(1, int(n_trials))

    best_labels = None
    best_info = None
    best_score = -np.inf

    for trial in range(n_trials):
        if start_idx_mode == "deterministic":
            start_idx = 0
        elif start_idx_mode == "random":
            start_idx = int(rng.integers(0, n))
        else:
            raise ValueError("start_idx_mode must be 'random' or 'deterministic'")

        seeds = farthest_point_sampling(
            points,
            k=n_test_sequences,
            start_idx=start_idx,
        )

        labels, info = build_split_from_seeds(
            points=points,
            seed_indices=seeds,
            min_dist=min_dist,
            test_window_size=test_window_size,
        )

        s = score_split(labels, alpha_none=alpha_none)

        # tie-breakers
        if (
            s > best_score
            or (
                np.isclose(s, best_score)
                and info["n_test"] > (best_info["n_test"] if best_info else -1)
            )
            or (
                np.isclose(s, best_score)
                and best_info is not None
                and info["n_test"] == best_info["n_test"]
                and info["n_none"] < best_info["n_none"]
            )
        ):
            best_score = s
            best_labels = labels
            best_info = dict(info)
            best_info["score"] = float(s)
            best_info["trial"] = trial
            best_info["start_idx"] = start_idx

    return best_labels, best_info


def split_trajectory_by_test_intervals(points, test_intervals, min_dist):
    """
    points: (N, 2) array of xy
    test_intervals: list of (start_idx, end_idx), inclusive, circular intervals allowed
    min_dist: minimum allowed distance between train and test
    
    Returns:
        labels: (N,) with values {TRAIN=0, TEST=1, NONE=-1}
    """
    def circular_interval_indices(n, start, end):
        """Inclusive circular interval [start, end] on a ring of length n."""
        if start <= end:
            return np.arange(start, end + 1)
        return np.concatenate([np.arange(start, n), np.arange(0, end + 1)])
    
    points = np.asarray(points, dtype=float)
    n = len(points)

    labels = np.full(n, TRAIN, dtype=int)

    # 1) mark requested test intervals
    test_mask = np.zeros(n, dtype=bool)
    for s, e in test_intervals:
        idx = circular_interval_indices(n, s % n, e % n)
        test_mask[idx] = True

    labels[test_mask] = TEST

    # 2) remove from train everything closer than min_dist to any test point
    tree_all = cKDTree(points)
    test_indices = np.where(test_mask)[0]

    conflict_mask = np.zeros(n, dtype=bool)
    for i in test_indices:
        nearby = tree_all.query_ball_point(points[i], r=min_dist)
        conflict_mask[nearby] = True

    # keep actual test points as TEST
    # points near test but not in test become NONE
    labels[(conflict_mask) & (~test_mask)] = NONE

    return labels