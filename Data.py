import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

n, m, t = 0, 0, 0
times = []
energies = []
jobs = []

def read_data(filename):
	"""
	leer datos del archivo csv
	"""
	global n, m, t, times, energies, jobs

	with open(filename, "r") as f:
		n, m, t = map(int, f.readline().split())

		times = []
		energies = []
		jobs = []

		for _ in range(n):
			row = list(map(float, f.readline().split()))
			times.append(row)

		for _ in range(n):
			row = list(map(float, f.readline().split()))
			energies.append(row)

		for _ in range(t):
			row = list(map(int, f.readline().split()))
			jobs.append( row)

	# print( n, m, t, times, energies, jobs)
	return n, m, t, times, energies, jobs


# 1-indexed
def get_time(op, machine):
    return times[op - 1][machine - 1]
# 1-indexed
def get_energy(op, machine):
    return energies[op - 1][machine - 1]
# calcula el tiempo y energia promedio de cada operacion, se puede modificar para maximo o minimo
def get_avg_time_energy():
    avg_time_job = []
    avg_energy_job = []

    for j, job in enumerate(jobs):
        op_means_time = []
        op_means_energy = []

        for op in job:
            mean_time = 0
            mean_energy = 0
            for machine in range(1, m + 1):
                mean_time += get_time(op, machine)
                mean_energy += get_energy(op, machine)
            mean_time /= m
            mean_energy /= m
            op_means_time.append(mean_time)
            op_means_energy.append(mean_energy)

        job_time = np.sum(op_means_time).item()
        job_energy = np.sum(op_means_energy).item()

        avg_time_job.append((job_time, j))
        avg_energy_job.append((job_energy, j))

    return avg_time_job, avg_energy_job

# ============================================
# orden -> arreglo que guarda las operaciones en el orden que deben ser procesadas
# last_idx -> arreglo auxiliar para implementar la restricion de las operaciones secuenciales

# preprocesamiento de FIFO,
def processing_FIFO(_):
    orden_fifo = [-1]
    last_idx_fifo = [-1]
    op2job = [-1]

    current_idx = 1
    for job_idx, job in enumerate(jobs):
        for i, op in enumerate(job):
            orden_fifo.append(op)
            if i == 0:
                last_idx_fifo.append(0)
            else:
                last_idx_fifo.append(current_idx - 1)
            current_idx += 1
            op2job.append(job_idx)

    return orden_fifo, last_idx_fifo, op2job

# preprocesamiento de LTP largest time processing,
def processing_LTP( avg_time_job):
    # Sort jobs by total processing time (descending → LTP)
    avg_time_job.sort(reverse=True, key=lambda x: x[0])

    orden_ltp = [-1]       # 1-indexed,
    last_idx_ltp = [-1]    # 1-indexed
    op2job = [-1]

    current_idx = 1        # 10indexed

    for _, job_idx in avg_time_job:
        job_ops = jobs[job_idx]

        for i, op in enumerate(job_ops):
            orden_ltp.append(op)

            if i == 0:
                last_idx_ltp.append(0)
            else:
                last_idx_ltp.append(current_idx - 1)

            current_idx += 1
            op2job.append(job_idx)

    return orden_ltp, last_idx_ltp, op2job

# preprocesamiento de STP shortest time processing
def processing_STP(avg_time_job):
    # Sort jobs by total processing time (descending → LTP)
    avg_time_job.sort( key=lambda x: x[0])

    orden_stp = [-1]       # 1-indexed,
    last_idx_stp = [-1]    # 1-indexed
    op2job = [-1]

    current_idx = 1        # 10indexed

    for _, job_idx in avg_time_job:
        job_ops = jobs[job_idx]

        for i, op in enumerate(job_ops):
            orden_stp.append(op)

            if i == 0:
                last_idx_stp.append(0)
            else:
                last_idx_stp.append(current_idx - 1)

            current_idx += 1
            op2job.append(job_idx)

    return orden_stp, last_idx_stp, op2job

# preprocesamiento de Round Robin FIFO,
def processing_round_robin_FIFO(_):
    orden_rr_fifo = [-1]
    last_idx_rr_fifo = [-1]
    op2job = [-1]

    indices = [0] * len(jobs)
    current_idx = 1
    max_sz = 0;
    for job in jobs:
        max_sz = max(max_sz, len(job))

    mtx_aux = []
    for job in jobs:
        mtx_aux.append(job + [0] * (max_sz - len(job)))

    for i in range(max_sz):
        for j in range(len(jobs)):
            if mtx_aux[j][i] != 0:
                orden_rr_fifo.append(mtx_aux[j][i])
                last_idx_rr_fifo.append(indices[j])
                indices[j] = current_idx
                current_idx += 1
                op2job.append(j)

    return orden_rr_fifo, last_idx_rr_fifo, op2job

# preprocesamiento de Round Robin LTP
def processing_round_robin_LTP(avg_time_job):
    avg_time_job.sort(reverse=True, key=lambda x: x[0])

    orden_rr_fifo = [-1]
    last_idx_rr_fifo = [-1]
    op2job = [-1]

    current_idx = 1

    indices = [0] * len(jobs)
    max_sz = 0;

    mtx_aux = []

    for job in jobs:
        max_sz = max(max_sz, len(job))

    for _, idx in avg_time_job:
        mtx_aux.append(jobs[idx] + [0] * (max_sz - len(jobs[idx])))

    for i in range(max_sz):
        for j in range(len(jobs)):
            if mtx_aux[j][i] != 0:
                orden_rr_fifo.append(mtx_aux[j][i])
                last_idx_rr_fifo.append(indices[j])
                indices[j] = current_idx
                current_idx += 1
                op2job.append(j)

    return orden_rr_fifo, last_idx_rr_fifo, op2job

# preprocesamiento de Round Robin ECA
def processing_round_robin_ECA(avg_energy_job):
    avg_energy_job.sort( key=lambda x: x[0])

    orden = [-1]
    last_idx = [-1]
    op2job = [-1]

    current_idx = 1

    indices = [0] * len(jobs)
    max_sz = 0;

    mtx_aux = []

    for job in jobs:
        max_sz = max(max_sz, len(job))

    for _, idx in avg_energy_job:
        mtx_aux.append(jobs[idx] + [0] * (max_sz - len(jobs[idx])))

    for i in range(max_sz):
        for j in range(len(jobs)):
            if mtx_aux[j][i] != 0:
                orden.append(mtx_aux[j][i])
                last_idx.append(indices[j])
                indices[j] = current_idx
                current_idx += 1
                op2job.append(j)

    return orden, last_idx, op2job


