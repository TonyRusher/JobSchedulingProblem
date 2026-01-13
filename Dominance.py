import numpy as np
import matplotlib.pyplot as plt
import Modulos

def paretoDominance_allFronts(points):
    """
    points: np.array of shape (n, 4) → [time, energy, ind_idx, policy_idx]
    Returns a list of Pareto fronts, each front as np.array with same 4 columns.
    Minimization problem (lower time, lower energy better).
    """
    points = np.array(points)
    sorted_points = points[np.argsort(points[:, 0])]
    pareto_frentes = []

    while len(sorted_points) > 0:
        front = []
        max_y = np.inf
        remaining = []

        for p in sorted_points:
            if p[1] < max_y:
                front.append(p)
                max_y = p[1]
            else:
                remaining.append(p)

        pareto_frentes.append(np.array(front))
        sorted_points = np.array(remaining)

    return pareto_frentes


def paretoDominance(points, N):
    """
    Same as above but stops once N points collected.
    Returns (n_points, fronts)
    """
    points = np.array(points)
    sorted_points = points[np.argsort(points[:, 0])]
    pareto_frentes = []
    n_indiv = 0

    while len(sorted_points) > 0:
        front = []
        max_y = np.inf
        remaining = []

        for p in sorted_points:
            if p[1] < max_y:
                front.append(p)
                max_y = p[1]
                n_indiv += 1
            else:
                remaining.append(p)

        pareto_frentes.append(np.array(front))
        sorted_points = np.array(remaining)

        if n_indiv >= N:
            return n_indiv, pareto_frentes

    return n_indiv, pareto_frentes


def pareto_first_front(points):
    """
    Compute the first Pareto front from full identity points.
    points: [time, energy, ind_idx, policy_idx]
    Returns np.array with same columns.
    """
    points = np.array(points)
    sorted_idx = np.argsort(points[:, 0])
    sorted_points = points[sorted_idx]

    front = []
    max_y = np.inf
    for p in sorted_points:
        if p[1] < max_y:
            front.append(p)
            max_y = p[1]

    return np.array(front)



def crowding_distance(points, debug=False):
    """
    points: np.array of shape (n, 4)
      Columns: [time, energy, ind_idx, policy_idx]
    Returns:
      same points sorted by descending distance, and distance array.
    """
    if len(points) == 0:
        return np.empty((0, 4)), np.array([])
    if len(points) == 1:
        return points, np.array([np.inf])

    distances = np.zeros(len(points))
    dim = 2  # time, energy

    for d in range(dim):
        sorted_idx = np.argsort(points[:, d])
        sorted_points = points[sorted_idx]

        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        denom = sorted_points[-1, d] - sorted_points[0, d]
        if denom == 0:
            continue

        for i in range(1, len(points) - 1):
            distances[sorted_idx[i]] += (
                (sorted_points[i + 1, d] - sorted_points[i - 1, d]) / denom
            )

    sorted_idx = np.argsort(-distances)
    sorted_points = points[sorted_idx]
    sorted_distances = distances[sorted_idx]

    if debug:
        for p, d in zip(sorted_points, sorted_distances):
            print(f"({p[0]:.2f}, {p[1]:.2f}) → Ind {int(p[2])}, Policy {int(p[3])}, Dist {d:.3f}")

    return sorted_points, sorted_distances

def hypervolume(front, reference=(500, 500)):
    """
    Compute hypervolume of 2D front w.r.t reference point (rectangular approximation).
    Assumes minimization.
    """
    front = front[np.argsort(front[:, 0])]
    hv = 0.0
    prev_x = reference[0]

    for p in front:
        width = prev_x - p[0]
        height = reference[1] - p[1]
        hv += width * height
        prev_x = p[0]

    return hv


def assign_fronts_and_crowding(population):
    """
    Given population, compute global Pareto fronts and crowding distances.
    Each point: [time, energy, idx_individual, idx_policy]
    Assigns front level and crowding distance per chromosome.
    """
    all_points = []
    for ind_idx, indiv in enumerate(population):
        for policy_idx, apt in enumerate(indiv.aptitudes):
            time, energy = apt[:2]
            all_points.append([time, energy, ind_idx, policy_idx])
    all_points = np.array(all_points)

    pareto_frentes = paretoDominance_allFronts(all_points)

    for indiv in population:
        indiv.front = [None] * indiv.n_chromosomes
        indiv.crowding_distance = [None] * indiv.n_chromosomes

    for front_level, front_points in enumerate(pareto_frentes, start=1):
        if len(front_points) == 0:
            continue

        sorted_front, distances = crowding_distance(front_points)

        for i, row in enumerate(sorted_front):
            ind_idx = int(row[2])
            policy_idx = int(row[3])
            population[ind_idx].front[policy_idx] = front_level
            population[ind_idx].crowding_distance[policy_idx] = distances[i]

    return population, pareto_frentes



def get_front_hypervolume(points,  reference=(100, 100)):
    points = np.array(points)
    front = pareto_first_front(points)
    hv = hypervolume(front, reference)
    return front, hv