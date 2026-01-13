import Data


# gracias al preprocesamiento solo se requiere una funcion de aptitud y los arreglos auxiliares de orden y last_idx
# defuelve tiempo, energia, schedule
# schedule es el arreglo que guarda el orden para graficarlos
def aptitud(chromosome, orden, last_idx):
    time = [0 for i in range(Data.m + 1)] #np.ceros(tam)
    out_time = [0 for i in range(len(chromosome))]
    energy = 0

    schedule = []
    # print(last_idx)

    for i in range(1,len(chromosome)):
        machine = chromosome[i]
        # print(out_time)
        # print(i)
        curr_time = max(out_time[last_idx[i]], time[machine]) # cuidado tony

        aux_time = Data.get_time(orden[i], machine)
        # time[machine] = max(time[machine], curr_time) + aux_time # cambie esto de abajo
        time[machine] =  curr_time + aux_time
        out_time[i] = time[machine]

        energy += Data.get_energy(orden[i], machine)


        schedule.append((machine, curr_time, aux_time, i))

    return max(time), energy, schedule