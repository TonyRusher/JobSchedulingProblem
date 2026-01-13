# Imprime un schedule dado
import numpy as np
import matplotlib.pyplot as plt
import Modulos
import Dominance
  # Carpeta donde se guardan las imagenes

def plot_points(points):
	plt.figure(fisgue=(8, 6))
	plt.scatter(points[:, 0], points[:, 1], c='blue', marker='o', alpha=0.5)

	plt.xlabel('Time')
	plt.ylabel('Energy')
	plt.grid(True)

	plt.show()

def plot_points_fronts(points, fronts):
    k_frentes = len(fronts)
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c='blue', marker='o', alpha=0.5, label='All Points')

    for i in range(k_frentes):
        print(f"size {i+1} = ", len(fronts[i]))
        plt.scatter(fronts[i][:, 0], fronts[i][:, 1],
                    c = Modulos.policy_colors[i % len(Modulos.policy_colors)], marker='x', label=f'P{i+1}')

    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Dominated points')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()  # Adjust layout so legend fits
    plt.show()
    
def plot_policy_front(points, color='blue', label='Policy', reference=(500, 500), show=True, semilla = 0, iter = 0, filename= "Escenario1.txt"):
    """
    Plot first front and compute hypervolume.
    points: Nx4 array [time, energy, ind_idx, policy_idx]
    """
    points = np.array(points)
    front = Dominance.pareto_first_front(points)

    hv = Dominance.hypervolume(front, reference)
    print(f"{label} → Hypervolume: {hv:.2f}, Points in first front: {len(front)}")

    # plt.scatter(points[:, 0], points[:, 1], c=color, marker='o', alpha=0.3, s=30, label = f'Dominados - {label}')
    # plt.scatter(front[:, 0], front[:, 1], c=color, marker='D', s=35, label=f'NO Dominados - {label}')
    plt.scatter(points[:, 0], points[:, 1], c=color, marker='o', alpha=0.3, s=30)
    plt.scatter(front[:, 0], front[:, 1], c=color, marker='D', s=35)

    if show:
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Energy', fontsize=12)
        plt.title('Dominated points', fontsize=14)
        plt.grid(True, alpha=0.3)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)

        plt.xlim(20, 100)
        plt.ylim(35, 100)
        plt.axis('equal')
        plt.gcf().set_size_inches(6, 6)

        plt.tight_layout()
        txt = "_" + str(semilla ) +"_points_" +  "iter_" + str(iter)
        plt.savefig(f"{filename}//{txt}.png")
        # plt.show()



    return front, hv






# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
def plott_schedule(schedule, m, op2job, policy_name="Policy", total_time=0, total_energy=0, semilla = 0, filename= "Escenario1.txt"):
    fig, ax = plt.subplots()
    idx_op = 1
    for machine, start, duration, label in schedule:
        job_id = op2job[idx_op]   # job/group for this operation
        color = Modulos.schedule_colors[job_id % len( Modulos.schedule_colors)]  # cycle through colors
        idx_op += 1

        ax.barh(
            y=machine,
            width=duration,
            left=start,
            height=0.4,
            align="center",
            color=color,
            edgecolor="black"
        )

        # Add text inside the bar
        ax.text(start + duration/2, machine, label,
                ha='center', va='center', color="white", fontsize=8)

    # Labeling
    ids_machines = [i for i in range(1,m)]
    ax.set_yticks(ids_machines)
    ax.set_yticklabels([f"Machine {i}" for i in ids_machines])
    ax.set_xlabel("Time")
    # ax.set_ylabel("Machines")
    ax.set_title(f"Schedule: {policy_name}", fontsize=12, fontweight='bold')
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    # Add info box outside the plot
    info_text = f"Time = {total_time:.2f}\nEnergy ={total_energy:.2f}"

    # Create a box with the information
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.text(1.02, 0.5, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props)

    plt.tight_layout()
    plt.savefig(f"{filename}/{semilla}_{policy_name}.png")
    # plt.show()
	
	
def plot_points(points):
  plt.figure(figsize=(8, 6))
  plt.scatter(points[:, 0], points[:, 1], c='blue', marker='o', alpha=0.5)

  plt.xlabel('Time')
  plt.ylabel('Energia')
  plt.grid(True)

  plt.show()

def plot_points_fronts(points, fronts):
    k_frentes = len(fronts)
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c='blue', marker='o', alpha=0.5, label='All Points')

    for i in range(k_frentes):
        print(f"size {i+1} = ", len(fronts[i]))
        plt.scatter(fronts[i][:, 0], fronts[i][:, 1],
                    c =  Modulos.policy_colors[i % len( Modulos.policy_colors)], marker='x', label=f'P{i+1}')

    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Non dominated and dominated points')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()  # Adjust layout so legend fits
    plt.show()


def plot_policy_front_app(ax, points, color, label, reference=(500, 500)):
    points = np.array(points)
    front = Dominance.pareto_first_front(points)

    ax.scatter(points[:, 0], points[:, 1], c=color, alpha=0.3, s=30)
    ax.scatter(front[:, 0], front[:, 1], c=color, marker='D', s=35)

    return front