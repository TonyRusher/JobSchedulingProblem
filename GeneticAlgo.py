import numpy as np
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt

import Aptitud, Data, Dominance, Individuo, Modulos, Plotting

crossover_values = [0.6, 0.8, 0.9]
mutation_values = [0.01, 0.1, 0.2]

def genetic_algorithm(n_generations=100,
                      population_size=20,
                      p_crossover=0.8,
                      p_mut_inter=0.3,
                      p_mut_exchange=0.2,
                      p_mut_displacement=0.1,
                      semilla = 998244353,
                      funciones = [],
                      orden = [],
					  last_idx = [],
                      funciones_name = [],
                      num_ops = 1,
					  from_app = False,
                      filename = "Escenario1.txt"
                    ):
  
    # --------------------- INITIALIZATION ---------------------
    population = []
    stored_plots = []

    for _ in range(population_size):
        ind = Individuo.Individuo(n_chromosomes=len(funciones))
        ind.build_random(num_ops)
        population.append(ind)

    # Initial evaluation
    for indiv in population:
        indiv.evaluate(orden, last_idx)

    hv_acum = [[1e18,0,0] for _ in range(len(funciones))] # [min, max, avg ] por cada politica
    hv_history = []
    # --------------------- GENERATIONS ---------------------
    prev_hv = None
    for gen in range(1, n_generations + 1):
        # ---- Pareto fronts and crowding distance ----
        population, _ = Dominance.assign_fronts_and_crowding(population)

        # ---- SELECTION & OFFSPRING ----
        offspring = []

        while len(offspring) < population_size:
            # Randomly pick two parents
            parentA, parentB = random.sample(population, 2)

            # Crossover
            if random.random() < p_crossover:
                child1, child2 = Individuo.crossover_uniform(parentA, parentB)
            else:
                child1, child2 = copy.deepcopy(parentA), copy.deepcopy(parentB)

            # Mutations
            if random.random() < p_mut_inter:
                child1.mutation_inter_chromosome()
                child2.mutation_inter_chromosome()

            child1.mutation_exchange(p=p_mut_exchange)
            child2.mutation_exchange(p=p_mut_exchange)

            child1.mutation_displacement(p=p_mut_displacement)
            child2.mutation_displacement(p=p_mut_displacement)

            # Evaluate children
            for child in [child1, child2]:
                child.aptitudes = []
                for i, chrom in enumerate(child.chromosomes):
                    fit = Aptitud.aptitud(chrom, orden[i], last_idx[i])
                    child.aptitudes.append(fit[:2])
                    child.schedules.append(fit[2])


            offspring.extend([child1, child2])

        # ---- MERGE AND SELECT NEXT GENERATION ----
        # Combine parents and offspring
        population += offspring
        population, _ = Dominance.assign_fronts_and_crowding(population)

        # Tournament selection to reduce back to population_size
        population = Individuo.tournament_selection(population)



        for policy_idx in range(len(funciones)):
            pointss = []
            for i, ind in enumerate(population):
                time, energy = ind.aptitudes[policy_idx][:2]
                pointss.append([time, energy, i, policy_idx])
            pointss = np.array(pointss)

            # plot Pareto front and get hypervolume
            front, hv = Dominance.get_front_hypervolume(pointss, reference=(500, 500))
            hv_acum[policy_idx][0] = min(hv_acum[policy_idx][0], hv)
            hv_acum[policy_idx][1] = max(hv_acum[policy_idx][1], hv)
            hv_acum[policy_idx][2] += hv
        
        # ---- Autoadaptative parameters every 10 generations ----
        
        current_hv = np.mean([
            Dominance.get_front_hypervolume(
                np.array([[*ind.aptitudes[i][:2], 0, i] for ind in population]),
                reference=(500, 500)
            )[1]
            for i in range(len(funciones))
        ])

        if gen % 10 == 0:
            if prev_hv is not None:
                if current_hv <= prev_hv:
                    p_mut_exchange = random.choice(mutation_values)
                    p_mut_displacement = random.choice(mutation_values)
                    p_crossover = random.choice(crossover_values[:-1])  # lower crossover
                else:
                    p_crossover = random.choice(crossover_values[1:])  # higher crossover
                    p_mut_exchange = random.choice(mutation_values[:-1])
                    p_mut_displacement = random.choice(mutation_values[:-1])

            prev_hv = current_hv

        # ---- HYPERVOLUME every 20 generations ----
        if gen % 20 == 0:
            row = [gen]
            for policy_idx in range(len(funciones)):
                min_hv, max_hv, avg_hv = hv_acum[policy_idx]
                avg_hv /= gen
                row += [int(min_hv), int(max_hv), int(avg_hv)]

            hv_history.append(row)


        if ((gen % 50 == 0 or gen == 0) and from_app) or gen == n_generations:
            print(f"=== Generation {gen} ===")
            # Collect first front of each policy
            fig, ax = plt.subplots(figsize=(4, 4))
            
            for policy_idx in range(len(funciones)):
                points = np.array([ind.aptitudes[policy_idx][:2] for ind in population])
                if not from_app:
                    Plotting.plot_policy_front(
                                    points,
                                    color=Modulos.policy_colors[policy_idx % len(Modulos.policy_colors)],
                                    label=funciones_name[policy_idx],
                                    reference=(500,500),
                                    show=False if policy_idx < len(funciones) - 1 else True,
                                    semilla = semilla,
                                    iter = gen,
                                    filename= filename
                            	)
                else:
                    Plotting.plot_policy_front_app(
											ax,
											points,
											color=Modulos.policy_colors[policy_idx % len(Modulos.policy_colors)],
											label=funciones_name[policy_idx]
                                        )
            if from_app:
                ax.set_xlabel("Time")
                ax.set_ylabel("Energy")
                ax.set_title(f"Pareto fronts – Gen {gen}")
                ax.grid(alpha=0.3)
                # ax.set_xlim(20,150)
                # ax.set_ylim(20,150)
                ax.set_aspect("equal", adjustable="box")
                ax.set_aspect("equal",adjustable="box")
                stored_plots.append({
                    "generation": gen,
                    "figure": fig
                })

    # Build column names dynamically
    cols = ['Generation']
    for name in funciones_name:
        cols += [f'{name}_min', f'{name}_max', f'{name}_avg']
    # df_hv = pd.DataFrame(hv_history, columns=cols)
    # print(df_hv.to_string(index=False))
    # df_hv.to_csv( f"{semilla}_hv_summary.csv", index=False)
    
    
    final_hv_stats = {}

    for i, name in enumerate(funciones_name):
        min_hv, max_hv, avg_hv = hv_acum[i]
        avg_hv /= n_generations
        final_hv_stats[name] = {
            "min": min_hv,
            "avg": avg_hv,
            "max": max_hv
        }
    
    
    if from_app:
        return population, stored_plots
    else:
        return population, final_hv_stats


