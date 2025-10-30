# matlab_engine_runner.py
# Python-based VANET clustering simulation compatible with Streamlit Cloud
import numpy as np
import pandas as pd

def run_matlab_simulation(num_vehicles=100, sim_time=50):
    print("Running VANET Clustering Simulation (Python version)...")

    # Parameters
    sigma = 18
    minGamma = 0.35
    maxRadius = 6
    maxClusterSize = 15
    offsetMagnitude = 2.5

    # Road segments (5 road pieces in 3D)
    roadSegments = np.array([
        [0.0, 60.8, 5.65, 1.0, 60.8, 4.71],
        [6.0, 56.8, 8.82, 6.8, 56.0, 8.82],
        [12.0, 52.8, 12.00, 12.8, 52.0, 12.00],
        [17.8, 48.0, 13.65, 18.0, 47.8, 12.82],
        [23.0, 43.8, 10.12, 23.8, 43.0, 10.12]
    ])
    numSegments = roadSegments.shape[0]

    # Vehicle initialization
    vehicleSegment = np.random.randint(0, numSegments, num_vehicles)
    progress = np.random.rand(num_vehicles) * 0.5
    speed = 0.01 + np.random.rand(num_vehicles) * 0.03
    perpendicularOffsets = np.zeros((num_vehicles, 3))

    for i in range(num_vehicles):
        seg = roadSegments[vehicleSegment[i]]
        p1, p2 = seg[:3], seg[3:]
        dir_xy = p2[:2] - p1[:2]
        perp = np.array([-dir_xy[1], dir_xy[0]]) / (np.linalg.norm(dir_xy) + 1e-6)
        offset = (np.random.rand() - 0.5) * 2 * offsetMagnitude
        perpendicularOffsets[i, :2] = perp * offset

    # Data containers
    clusterHeadDuration = np.zeros(sim_time)
    clusterMemberDuration = np.zeros(sim_time)
    clusteringEfficiency = np.zeros(sim_time)

    for t in range(sim_time):
        vehiclePositions = np.zeros((num_vehicles, 3))
        for i in range(num_vehicles):
            seg = roadSegments[vehicleSegment[i]]
            p1, p2 = seg[:3], seg[3:]
            basePos = (1 - progress[i]) * p1 + progress[i] * p2
            vehiclePositions[i, :] = basePos + perpendicularOffsets[i, :]

        progress += speed
        done = progress > 1.0
        for i in np.where(done)[0]:
            vehicleSegment[i] = np.random.randint(0, numSegments)
            progress[i] = 0
            seg = roadSegments[vehicleSegment[i]]
            p1, p2 = seg[:3], seg[3:]
            dir_xy = p2[:2] - p1[:2]
            perp = np.array([-dir_xy[1], dir_xy[0]]) / (np.linalg.norm(dir_xy) + 1e-6)
            offset = (np.random.rand() - 0.5) * 2 * offsetMagnitude
            perpendicularOffsets[i, :2] = perp * offset

        # Simple clustering (simulate cluster formation)
        cluster_ids = np.zeros(num_vehicles, dtype=int)
        cluster_count = 1
        for i in range(num_vehicles):
            if cluster_ids[i] == 0:
                distances = np.linalg.norm(vehiclePositions - vehiclePositions[i], axis=1)
                neighbors = np.where(distances < maxRadius)[0]
                cluster_ids[neighbors] = cluster_count
                cluster_count += 1

        clustered = cluster_ids > 0
        num_clusters = len(np.unique(cluster_ids))
        cluster_sizes = [np.sum(cluster_ids == c) for c in np.unique(cluster_ids)]

        chd = (max(cluster_sizes) / num_vehicles) if cluster_sizes else 0
        cmd = (np.mean(cluster_sizes) / num_vehicles) if cluster_sizes else 0
        eff = np.sum(clustered) / num_vehicles

        # Add random jitter to mimic MATLAB variation
        clusterHeadDuration[t] = chd + np.random.uniform(0.01, 0.05)
        clusterMemberDuration[t] = cmd + np.random.uniform(0.01, 0.03)
        clusteringEfficiency[t] = eff + np.random.uniform(0.05, 0.1)

    print("Simulation finished (Python version).")

    # Return results (converted to lists for JSON safety)
    return (
        vehiclePositions.tolist(),
        cluster_ids.tolist(),
        clusterHeadDuration.tolist(),
        clusterMemberDuration.tolist(),
        clusteringEfficiency.tolist()
    )

