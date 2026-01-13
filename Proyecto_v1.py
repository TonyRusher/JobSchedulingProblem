import Data
import GeneticAlgo
import Dominance

import numpy as np
import random
import pandas as pd


# --------------------- READ DATA ---------------------
filename = "Escenario3"
n, m, t, times, energies, jobs = Data.read_data(filename + ".txt")

avg_time_job, avg_energy_job = Data.get_avg_time_energy()

# --------------------- PROCESS ORDERS ---------------------
funciones = [
    Data.processing_FIFO,
    Data.processing_LTP,
    Data.processing_STP,
    Data.processing_round_robin_FIFO,
    Data.processing_round_robin_LTP,
    Data.processing_round_robin_ECA
]

funciones_name = ["FIFO", "LTP", "STP", "RR FIFO", "RR LTP", "RR ECA"]

orden, last_idx, op2job = [], [], []
num_ops = sum(len(job) for job in jobs) + 1

for f in funciones:
    o, l, o2j = f(avg_time_job)
    orden.append(o)
    last_idx.append(l)
    op2job.append(o2j)

# --------------------- SEEDS ---------------------
seeds = [
    998244353, 1000000007, 1000000009,
    1000000033, 1000000093, 1000000097,
    179424691, 32452843, 49979687, 67867967
]

all_seed_results = []

# --------------------- RUN EXPERIMENTS ---------------------
for idx, sem in enumerate(seeds):
    random.seed(sem)
    print(f"=============== Seed: {sem} ===============")

    population, hv_stats = GeneticAlgo.genetic_algorithm(
        n_generations=200,
        population_size=20,
        p_crossover=0.8,
        p_mut_inter=0.3,
        p_mut_exchange=0.2,
        p_mut_displacement=0.1,
        semilla=idx,                  
        funciones=funciones,
        orden=orden,
        last_idx=last_idx,
        funciones_name=funciones_name,
        num_ops=num_ops,
        from_app=False,
        filename=filename
    )

    hv_stats["seed"] = sem
    all_seed_results.append(hv_stats)

    # -------- OPTIONAL: final Pareto plots per seed --------
    front_ans = []
    hv_ans = []

    for policy_idx in range(len(funciones)):
        points = np.array([
            [*ind.aptitudes[policy_idx][:2], i, policy_idx]
            for i, ind in enumerate(population)
        ])

        front, hv = Dominance.get_front_hypervolume(points, reference=(500, 500))
        front_ans.append(front)
        hv_ans.append(hv)

    for policy_idx, front in enumerate(front_ans):
        if len(front) == 0:
            continue

        mid_idx = len(front) // 2
        ind_idx = int(front[mid_idx][2])
        pol_idx = int(front[mid_idx][3])

        ind = population[ind_idx]
        ind.plot(
            orden,
            last_idx,
            pol_idx,
            idx,
            op2job,
            funciones_name,
            filename=filename
        )

# --------------------- AGGREGATE ACROSS SEEDS ---------------------
rows = []

for result in all_seed_results:
    seed = result["seed"]
    for policy, stats in result.items():
        if policy == "seed":
            continue
        rows.append({
            "seed": seed,
            "policy": policy,
            "min": stats["min"],
            "avg": stats["avg"],
            "max": stats["max"]
        })

df = pd.DataFrame(rows)

df_summary = (
    df.groupby("policy")
      .agg(
          min_over_seeds=("min", "min"),
          avg_over_seeds=("avg", "mean"),
          max_over_seeds=("max", "max")
      )
      .reset_index()
)

df_summary.to_csv(f"{filename}hv_summary_across_seeds.csv", index=False)

