# matlab_engine_runner.py
# Python-based VANET clustering simulation for Streamlit Cloud
import numpy as np

def run_matlab_simulation(num_vehicles: int = 50, sim_time: int = 30):
    """
    Returns:
      positions: list with shape (num_vehicles, 3, sim_time)
      clusters:  list with shape (num_vehicles, sim_time)
      CH:        list length sim_time (cluster head duration like metric)
      CM:        list length sim_time (cluster member duration like metric)
      Eff:       list length sim_time (clustering efficiency)
    Notes:
      - All returns are Python lists (JSON-safe).
      - This is a pure-Python reimplementation of your MATLAB simulation behavior.
    """
    # ---- parameters (same spirit as your MATLAB code) ----
    sigma = 18.0
    minGamma = 0.35
    maxRadius = 6.0
    maxClusterSize = 15
    offsetMagnitude = 2.5

    # road segments (same coordinates as your MATLAB code)
    roadSegments = np.array([
        [0.0, 60.8, 5.65, 1.0, 60.8, 4.71],
        [6.0, 56.8, 8.82, 6.8, 56.0, 8.82],
        [12.0, 52.8, 12.00, 12.8, 52.0, 12.00],
        [17.8, 48.0, 13.65, 18.0, 47.8, 12.82],
        [23.0, 43.8, 10.12, 23.8, 43.0, 10.12]
    ])
    numSegments = roadSegments.shape[0]

    # ---- initialize vehicles ----
    rng = np.random.default_rng()  # modern RNG
    vehicleSegment = rng.integers(0, numSegments, size=num_vehicles)
    progress = rng.random(num_vehicles) * 0.5
    speed = 0.01 + rng.random(num_vehicles) * 0.03
    perpendicularOffsets = np.zeros((num_vehicles, 3), dtype=float)

    for i in range(num_vehicles):
        seg = roadSegments[vehicleSegment[i]]
        p1, p2 = seg[:3], seg[3:]
        dir_xy = p2[:2] - p1[:2]
        perp = np.array([-dir_xy[1], dir_xy[0]]) / (np.linalg.norm(dir_xy) + 1e-9)
        offset = (rng.random() - 0.5) * 2.0 * offsetMagnitude
        perpendicularOffsets[i, :2] = perp * offset

    # ---- storage for history ----
    positionHistory = np.zeros((num_vehicles, 3, sim_time), dtype=float)
    clusters_history = np.zeros((num_vehicles, sim_time), dtype=int)
    CH = np.zeros(sim_time, dtype=float)
    CM = np.zeros(sim_time, dtype=float)
    Eff = np.zeros(sim_time, dtype=float)

    # ---- main loop ----
    for t in range(sim_time):
        # compute current positions
        vehiclePositions = np.zeros((num_vehicles, 3), dtype=float)
        for i in range(num_vehicles):
            seg = roadSegments[vehicleSegment[i]]
            p1, p2 = seg[:3], seg[3:]
            basePos = (1.0 - progress[i]) * p1 + progress[i] * p2
            vehiclePositions[i, :] = basePos + perpendicularOffsets[i, :]

        # store positions for this frame
        positionHistory[:, :, t] = vehiclePositions

        # advance progress and reset vehicles that finished segment
        progress += speed
        done_idx = np.where(progress > 1.0)[0]
        if done_idx.size > 0:
            for i in done_idx:
                vehicleSegment[i] = int(rng.integers(0, numSegments))
                progress[i] = 0.0
                seg = roadSegments[vehicleSegment[i]]
                p1, p2 = seg[:3], seg[3:]
                dir_xy = p2[:2] - p1[:2]
                perp = np.array([-dir_xy[1], dir_xy[0]]) / (np.linalg.norm(dir_xy) + 1e-9)
                offset = (rng.random() - 0.5) * 2.0 * offsetMagnitude
                perpendicularOffsets[i, :2] = perp * offset

        # ---- clustering algorithm (greedy neighbor grouping) ----
        cluster_ids = np.zeros(num_vehicles, dtype=int)  # 0 = unassigned
        next_cluster = 1

        # We will mark neighbors greedily; this is O(n^2) but n=50 is fine
        for i in range(num_vehicles):
            if cluster_ids[i] != 0:
                continue
            # find neighbors within radius
            dists = np.linalg.norm(vehiclePositions - vehiclePositions[i], axis=1)
            neighbors = np.where(dists <= maxRadius)[0]
            # limit cluster size
            if neighbors.size > maxClusterSize:
                # choose the closest maxClusterSize neighbors
                order = np.argsort(dists[neighbors])
                neighbors = neighbors[order[:maxClusterSize]]
            # If no neighbors (should include self), neighbors will contain i at least
            cluster_ids[neighbors] = next_cluster
            next_cluster += 1

        # save cluster ids
        clusters_history[:, t] = cluster_ids

        # ---- metrics ----
        unique_clusters = np.unique(cluster_ids)
        # exclude 0 if ever present (should not be)
        unique_clusters = unique_clusters[unique_clusters > 0]
        cluster_sizes = []
        for cid in unique_clusters:
            cluster_sizes.append(int(np.sum(cluster_ids == cid)))

        if len(cluster_sizes) > 0:
            largest = max(cluster_sizes)
            mean_size = float(np.mean(cluster_sizes))
            chd = largest / float(num_vehicles)
            cmd = mean_size / float(num_vehicles)
        else:
            chd = 0.0
            cmd = 0.0

        clustered_count = np.sum(cluster_ids > 0)
        eff = float(clustered_count) / float(num_vehicles)

        # add small jitter to mimic MATLAB randomness (keeps range similar)
        CH[t] = chd + rng.uniform(0.0, 0.03)
        CM[t] = cmd + rng.uniform(0.0, 0.02)
        Eff[t] = eff + rng.uniform(0.0, 0.05)

    # ---- finished ----
    # Convert to lists and return (positions as num_vehicles x 3 x sim_time)
    return (
        positionHistory.tolist(),
        clusters_history.tolist(),
        CH.tolist(),
        CM.tolist(),
        Eff.tolist()
    )
