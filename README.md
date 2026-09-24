# Allerton 2026 RIS-assisted cell-free massive MIMO code

Code and saved simulation data for:

> Mohammad Ezaz and Azadeh Vosoughi, "Energy-Efficient RIS-Assisted User-Centric Cell-Free Massive MIMO: A Game-Theoretic Framework," 62nd Allerton Conference on Communication, Control, and Computing, 2026.

The paper studies pilot assignment, AP-user association, and downlink power allocation in a user-centric cell-free system with and without one fixed-phase RIS. The repository supports two related tasks: regenerating the paper's figures and tables from the saved run data, and executing new simulations with the same recorded settings.

## Recreate the paper figures and tables

The six `paper_data/` folders contain the `merged_data.npz` files used for the paper. From this repository's top-level folder, run:

```bash
python process_convergence_results.py
```

This step needs NumPy and Matplotlib. It does not need CVXPY and does not rerun the simulations. Output is placed in a timestamped folder under `figures/`. The `tables/` subfolder contains `paper_association_table.csv` and `paper_power_table.csv`, which can be compared with the original CSV files under `paper_data/reference_tables/`.

| Paper item | Generated file |
| --- | --- |
| Figure 2, association convergence | `pdf/fig1_association_normalized_side_by_side.pdf` |
| Figure 3, power-allocation convergence | `pdf/fig2_power_dinkelbach_normalized_side_by_side.pdf` |
| Table I, association initialization | `tables/paper_association_table.csv` |
| Table II, power allocation | `tables/paper_power_table.csv` |

The script also creates individual plots and detailed tables. Figure numbers in its filenames reflect the order of generated plots, so they differ from the figure numbers in the paper. Figure 1 of the paper is the system diagram.

## Re-run the simulations

The original experiments used Python 3.11.15. Use Python 3.11 for a fresh run and install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

The downlink power stage is configured to try `CLARABEL` and then `SCS`. Confirm that both appear in the printed solver list before running it. The code records both converged runs and runs that stop after rejecting a candidate; inspect the generated summaries before using newly generated results.

To check one setup without altering the saved paper data:

```bash
python run_paper_experiments.py final_tauK_random_S200 --smoke
```

The six original experiments each consist of 100 chunks. For example, start the first experiment with:

```bash
python run_paper_experiments.py final_tauK_random_S200 --chunk-id 0
```

Run the same command for **every remaining chunk ID from `1` through `99`**, replacing `0` with that ID. Once all 100 chunks have finished, merge them:

```bash
python run_paper_experiments.py final_tauK_random_S200 --merge
```

The complete runs can take substantial compute time. Each chunk writes to its own file under `generated_results/full/<experiment_name>/chunks/`. Smoke runs write to `generated_results/smoke/`. These folders are excluded from Git.

| Experiment name | Pilots | Initialization | Setups |
| --- | --- | --- | ---: |
| `final_tauK_random_S200` | `tau_p = K` | Random | 200 |
| `final_tauKhalf_random_S200` | `tau_p = K/2` | Random | 200 |
| `sen_tauK_strongest_S100` | `tau_p = K` | Strongest singleton | 100 |
| `sen_tauK_full_S100` | `tau_p = K` | Full candidate | 100 |
| `sen_tauKhalf_strongest_S100` | `tau_p = K/2` | Strongest singleton | 100 |
| `sen_tauKhalf_full_S100` | `tau_p = K/2` | Full candidate | 100 |

The launcher uses the seed and settings recorded with the original six runs. The scenario code runs both no-RIS and single-RIS cases, with 10 users, 35 four-antenna APs, and a 100-element RIS in the single-RIS case. The RIS phase is fixed at 45 degrees; the code does not optimize RIS phases.


## Citation

If this code helps your work, please cite the paper listed above.
