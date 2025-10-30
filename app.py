# app.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import time
from matlab_engine_runner import run_matlab_simulation

st.set_page_config(page_title="VANET 3D Dashboard", layout="wide")

st.title("🚗 Real-Time 3D Gamma-Based VANET Clustering Dashboard")

num_vehicles = st.sidebar.slider("Number of Vehicles", 10, 200, 50)
sim_time = st.sidebar.slider("Simulation Duration", 10, 100, 30)
start_btn = st.sidebar.button("Run MATLAB Simulation")

if start_btn:
    st.info("Running MATLAB Simulation... Please wait ⏳")
    pos, clusters, CH, CM, Eff = run_matlab_simulation(num_vehicles, sim_time)
    st.success("Simulation Complete ✅")

    placeholder3D = st.empty()
    metric_chart = st.empty()

    metrics = pd.DataFrame({"CH": CH, "CM": CM, "Eff": Eff})

    for t in range(sim_time):
        frame_positions = np.array(pos)
        frame_clusters = clusters[:, t]

        df = pd.DataFrame(frame_positions, columns=["x", "y", "z"])
        df["cluster"] = frame_clusters

        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="cluster",
            color_continuous_scale="Turbo",
            title=f"Frame {t+1}/{sim_time}"
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        placeholder3D.plotly_chart(fig, use_container_width=True)

        metric_chart.line_chart(metrics.iloc[:t+1])
        time.sleep(0.3)
