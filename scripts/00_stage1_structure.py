"""
Step 00 (optional, --full path) - Stage-1 PSO structure optimization (Fig. 7).

Runs the expensive PSO architecture search for LSTM and CNN and plots the
RMSE-vs-iteration convergence (Fig. 7).  The discovered architectures are
printed for comparison with the paper's Table 2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import plotting as P
from src.optimize_structure import optimize_structure


def main(n_iter=40, n_particles=10, epochs=18):
    histories = {}
    for kind in ["LSTM", "CNN"]:
        print(f"Stage-1 PSO structure search for {kind} ...")
        hp, rmse, hist = optimize_structure(kind, n_particles=n_particles,
                                            n_iter=n_iter, epochs=epochs)
        histories[kind] = hist
        print(f"  best RMSE={rmse:.4f}  nodes={hp['nodes']}  "
              f"dropout={hp.get('dropout', hp.get('dropout'))}")
    P.fig_convergence(histories,
                      "Stage-1 structure optimization with PSO (Fig. 7)",
                      "fig07_stage1_convergence.png")
    print("Saved Fig. 7.")


if __name__ == "__main__":
    main()
