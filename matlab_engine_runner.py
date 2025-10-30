# matlab_engine_runner.py
import matlab.engine
import numpy as np
import pandas as pd

def run_matlab_simulation(num_vehicles=50, sim_time=50):
    print("Starting MATLAB Engine...")
    eng = matlab.engine.start_matlab()
    eng.cd(r"C:\Program Files\MATLAB\R2025b\extern\engines\python", nargout=0)  # set path to where Gamma_VANET.m is saved

    print("Running Gamma_VANET in MATLAB...")
    positions, clusters, CH, CM, Eff = eng.Gamma_VANET(num_vehicles, sim_time, nargout=5)

    # Convert MATLAB arrays to numpy
    positions = np.array(positions)
    clusters = np.array(clusters)
    CH = np.array(CH).flatten()
    CM = np.array(CM).flatten()
    Eff = np.array(Eff).flatten()

    print("Simulation finished.")
    return positions, clusters, CH, CM, Eff
