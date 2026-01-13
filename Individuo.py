import random
import copy

import Aptitud
import Data, Plotting


class Individuo:
    def __init__(self, n_chromosomes=6):
        self.n_chromosomes = n_chromosomes

        self.chromosomes = []              # list of chromosomes
        self.aptitudes = []                # list of lists: [(time, energy), ...] per chromosome/policy
        self.schedules = []                # schedules separately
        self.front = []                    # Pareto front level per chromosome
        self.crowding_distance = []        # Crowding distance per chromosome

    # ------------------ BUILD ------------------
    def build_random(self, num_ops):
        """Random permutation for each chromosome."""
        self.chromosomes = []
        for _ in range(self.n_chromosomes):
            chrom = [0 for i in range(num_ops)]
            for i in range(1, num_ops):
                chrom[i] = random.randint(1, Data.m)
            self.chromosomes.append(chrom)
    # ------------------ EVALUATION ------------------
    def evaluate(self, orden_list, last_idx_list):
        """
        Evaluate all chromosomes across all policies.
        - orden_list, last_idx_list: lists of length n_chromosomes
        """
        self.aptitudes = []
        self.schedules = []

        for idx, chrom in enumerate(self.chromosomes):
            orden = orden_list[idx]
            last_idx = last_idx_list[idx]
            # print(chrom)
            # print(orden)
            # print(last_idx)
            time, energy, schedule = Aptitud.aptitud(chrom, orden, last_idx)
            self.schedules.append(schedule)

            self.aptitudes.append((time, energy))
        # print(self.aptitudes)
        return self.aptitudes

    # ------------------ FRONT / CROWDING ------------------
    def set_front_and_crowding(self, fronts, distances):
        """Assign Pareto front and crowding distances per chromosome."""
        self.front = list(fronts)
        self.crowding_distance = list(distances)

    # ------------------ MUTATIONS ------------------
    def mutation_inter_chromosome(self):
        i, j = random.sample(range(1, self.n_chromosomes), 2)
        self.chromosomes[i], self.chromosomes[j] = self.chromosomes[j], self.chromosomes[i]

    def mutation_exchange(self, p=0.1):
        for chrom in self.chromosomes:
            if random.random() < p:
                i, j = random.sample(range(1, len(chrom)), 2)  # ← Start from 1, not 0!
                chrom[i], chrom[j] = chrom[j], chrom[i]

    def mutation_displacement(self, p=0.1):
        for idx, chrom in enumerate(self.chromosomes):
            if random.random() < p:
                n = len(chrom) - 1  # Exclude position 0
                i, j, k = sorted(random.sample(range(1, len(chrom)), 3))

                segment = chrom[i:j]
                rest = chrom[1:i] + chrom[j:]  # ← Start from 1, not 0!
                k = (k - 1) % len(rest)  # Adjust k since we're working with rest[1:]

                chrom_new = [0] + rest[:k] + segment + rest[k:]  # ← Explicitly keep 0 at front
                self.chromosomes[idx] = chrom_new

    # ------------------ PLOTTING ------------------
    def plot(self, orden_list, last_idx_list, policy, semilla, op2job, funciones_name, filename= "Escenario1.txt"):
        """Plot all schedules if available."""
        schedule = self.schedules[policy]
        time, energy = self.aptitudes[policy]
        print(f"[Chromosome {policy}] Tiempo: {time:.2f} | Energía: {energy:.2f}")
        Plotting.plott_schedule(schedule, Data.m + 1, op2job[policy], funciones_name[policy], time, energy, semilla, filename= filename)

    # ------------------ REPRESENTATION ------------------
    def __repr__(self):
        return (f"<Individuo chromosomes={len(self.chromosomes)}, "
                f"aptitudes={len(self.aptitudes)}, "
                f"fronts={len(self.front)}, "
                f"crowding={len(self.crowding_distance)}, "
                f"schedules={len(self.schedules)}>")


def crossover_uniform(parentA, parentB):
    assert parentA.n_chromosomes == parentB.n_chromosomes, "Parent mismatch in chromosome count"

    n_chromosomes = parentA.n_chromosomes
    num_ops = len(parentA.chromosomes[0])

    child1 = Individuo(n_chromosomes)
    child2 = Individuo(n_chromosomes)

    child1.chromosomes = []
    child2.chromosomes = []

    for c in range(n_chromosomes):
        chromA = parentA.chromosomes[c]
        chromB = parentB.chromosomes[c]

        # Start with dummy 0 at position 0
        child1_chrom = [0]
        child2_chrom = [0]

        # Uniform crossover per gene (skip index 0)
        for i in range(1, num_ops):  # ← Changed from range(num_ops)
            if random.random() < 0.5:
                child1_chrom.append(chromA[i])
                child2_chrom.append(chromB[i])
            else:
                child1_chrom.append(chromB[i])
                child2_chrom.append(chromA[i])

        child1.chromosomes.append(child1_chrom)
        child2.chromosomes.append(child2_chrom)

    return child1, child2
	
import random

def merge_selection(A, B):
    """
    Creates a new individual by selecting the best chromosome between
    two individuals A and B based on Pareto front and crowding distance.
    """
    assert len(A.chromosomes) == len(B.chromosomes), "Individuals must have same number of chromosomes"

    n_chromosomes = len(A.chromosomes)
    child = copy.deepcopy(A)       # start with a copy of A’s structure
    child.chromosomes = []         # will rebuild
    child.aptitudes = []
    child.front = []
    child.crowding_distance = []

    for i in range(n_chromosomes):
        # compare chromosome i from A and B
        front_a, front_b = A.front[i], B.front[i]
        crowd_a, crowd_b = A.crowding_distance[i], B.crowding_distance[i]

        if front_a < front_b:
            chosen = (A.chromosomes[i], A.aptitudes[i], front_a, crowd_a)
        elif front_b < front_a:
            chosen = (B.chromosomes[i], B.aptitudes[i], front_b, crowd_b)
        else:  # same front, pick the one with larger crowding
            if crowd_a >= crowd_b:
                chosen = (A.chromosomes[i], A.aptitudes[i], front_a, crowd_a)
            else:
                chosen = (B.chromosomes[i], B.aptitudes[i], front_b, crowd_b)

        child.chromosomes.append(chosen[0])
        child.aptitudes.append(chosen[1])
        child.front.append(chosen[2])
        child.crowding_distance.append(chosen[3])

    return child


def tournament_selection(population, consecutive=True):
    """
    Select the next generation by pairing individuals.
    If consecutive=True, pairs are (p1,p2), (p3,p4), ...
    Otherwise, random pairs are used.
    """
    next_generation = []
    n = len(population)

    if consecutive:
        # ensure even number of individuals
        if n % 2 != 0:
            population = population[:-1]

        for i in range(0, n, 2):
            A = population[i]
            B = population[i + 1]
            next_generation.append(merge_selection(A, B))

    else:
        # original random selection
        for _ in range(n // 2):
            A, B = random.sample(population, 2)
            next_generation.append(merge_selection(A, B))

    return next_generation


