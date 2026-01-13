import streamlit as st
import random

import Data
import GeneticAlgo


# -------------------- SESSION STATE --------------------
if "plots" not in st.session_state:
    st.session_state.plots = None


# -------------------- PAGE CONFIG --------------------
st.set_page_config(layout="wide")
st.title("Genetic Algorithm Scheduler")


# -------------------- LAYOUT --------------------
left, right = st.columns([1, 3])  # controls | plots


# -------------------- LEFT: CONTROLS --------------------
with left:
    st.header("Controls")

    uploaded_file = st.file_uploader(
        "Upload scenario file",
        type=["txt"]
    )

    seed = st.number_input(
        "Seed",
        value=998244353,
        step=1
    )

    n_generations = st.number_input(
        "Number of generations",
        value=200,
        step=10
    )

    run = st.button("Run algorithm")


# -------------------- RUN ALGORITHM --------------------
if run and uploaded_file is not None:
    random.seed(seed)

    # Save uploaded file temporarily
    with open("temp_scenario.txt", "wb") as f:
        f.write(uploaded_file.read())

    # -------- READ DATA --------
    n, m, t, times, energies, jobs = Data.read_data("temp_scenario.txt")
    avg_time_job, avg_energy_job = Data.get_avg_time_energy()

    # -------- POLICIES --------
    funciones = [
        Data.processing_FIFO,
        Data.processing_LTP,
        Data.processing_STP,
        Data.processing_round_robin_FIFO,
        Data.processing_round_robin_LTP,
        Data.processing_round_robin_ECA
    ]

    funciones_name = [
        "FIFO", "LTP", "STP",
        "RR FIFO", "RR LTP", "RR ECA"
    ]

    orden, last_idx, op2job = [], [], []
    num_ops = sum(len(job) for job in jobs) + 1

    for f in funciones:
        o, l, o2j = f(avg_time_job)
        orden.append(o)
        last_idx.append(l)
        op2job.append(o2j)

    # -------- RUN GA --------
    population, plots = GeneticAlgo.genetic_algorithm(
        n_generations=n_generations,
        population_size=20,
        p_crossover=0.8,
        p_mut_inter=0.3,
        p_mut_exchange=0.2,
        p_mut_displacement=0.1,
        semilla=seed,
        funciones=funciones,
        orden=orden,
        last_idx=last_idx,
        funciones_name=funciones_name,
        num_ops=num_ops,
        from_app=True
    )

    # -------- STORE RESULTS --------
    st.session_state.plots = plots


# -------------------- RIGHT: PLOTS --------------------
with right:
    st.header("Pareto Front Evolution")

    if st.session_state.plots is not None:
        plots = st.session_state.plots

        gen_idx = st.slider(
            "Generation",
            min_value=0,
            max_value=len(plots) - 1,
            value=0
        )

        gen_data = plots[gen_idx]

        st.write(f"Generation {gen_data['generation']}")
        st.pyplot(gen_data["figure"], width='stretch')

    else:
        st.info("Run the algorithm to visualize results.")
