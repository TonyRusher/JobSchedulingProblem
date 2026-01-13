# Job Scheduling Problem – NSGA-II Bio-Inspired Optimization

This repository contains a bio-inspired optimization project focused on solving a **Job Scheduling Problem (JSP)** using the **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**. The project includes:

* A main execution script where different **scenarios** can be introduced
* An **NSGA-II implementation** tailored to the job scheduling problem
* A simple **application/interface script** to run experiments
* Support for multiple scenarios with different numbers of jobs, machines, and operations

The goal of the project is to approximate the **Pareto front** for multiple conflicting objectives (e.g., makespan, machine utilization, etc.).

---
## ⚙️ Requirements

* Python **3.8+**
* streamlit (for running the app)

---


## ▶️ How to Run the Project

### 1️⃣ Select a Scenario
Open `Proyecto_v1.py`. You will find a variable similar to:

```python
filename = "Escenario1"
```

Change this value to select the desired scenario:

* `Escenario1` – small instance (few jobs and machines)
* `Escenario2` – medium instance
* `Escenario3` – large instance

Each scenario defines:

* Number of jobs
* Number of machines
* Number of operations per job

---

### 2️⃣ Run the Main File

From the project root:

```bash
python Proyecto_v1.py
```

This will:

* Load the selected scenario
* Initialize the NSGA-II population
* Execute the evolutionary process
* Output the final Pareto front and metrics

---

## 🧬 NSGA-II Overview (Project Context)

The algorithm follows the classical NSGA-II workflow:

1. Population initialization
2. Fitness evaluation (multi-objective)
3. Non-dominated sorting
4. Crowding distance assignment
5. Selection, crossover, and mutation
6. Elitist replacement

The output is a set of **non-dominated solutions** representing trade-offs between objectives.

---

## 🖥️ Running the Application Script

If you want to execute the experimental application:

```bash
streamlit run app.py  
```

The application script is intended for:

* Running multiple executions
* Testing different seeds
* Comparing scenarios
* Collecting performance metrics


## 👤 Author

**Jorge Antonio Juárez Leyva**

Academic project – Bio-Inspired Optimization
