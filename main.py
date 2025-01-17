#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main execution script for the Newton-Girard trace estimation experiment.

Key points:
- We handle 'gaps' as a list. For random distribution, we ignore them ([-1]).
- For arithmetic/geometric, we do a triple nested loop over ranks, k_values, and gaps.
- The CSV's first column is a running integer index, starting from 0, incremented each row.

If gen_arithmetic fails, it should print an error like:
  [gen_arithmetic] Error: Cannot satisfy max_min_gap=... for rank=...
"""

import os
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import config
from utils import (
    gen_arithmetic, gen_geometric, gen_random,
    sum_with_error, cal_coefficient, cal_power_trace
)


def main():
    # 1) Create 'result/' folder
    os.makedirs('result', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('result', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 2) Possibly modify config based on distribution
    distribution = config.distribution.lower().strip()

    if distribution == "random":
        # If any gap != -1, warn and set them to [-1]
        if config.gaps != [-1]:
            print(
                f"[main] Warning: distribution='random' but gaps={config.gaps} != [-1]. Setting gaps = [-1].")
            config.gaps = [-1]
    else:
        # distribution in ["arithmetic", "geometric"]
        if config.iteration != -1:
            print(
                f"[main] Warning: distribution='{distribution}', iteration={config.iteration} != -1. Setting iteration = -1.")
            config.iteration = -1

    # 3) Copy config.py to the result folder (instead of saving config.json)
    try:
        with open("config.py", "r", encoding="utf-8") as f_in:
            config_py_content = f_in.read()
        with open(os.path.join(save_dir, "config.py"), "w", encoding="utf-8") as f_out:
            f_out.write(config_py_content)
    except FileNotFoundError:
        print("[main] Warning: config.py not found. Skipped copying.")

    # 4) Prepare CSV for results
    csv_path = os.path.join(save_dir, "results.csv")
    f_csv = open(csv_path, 'w', newline='', encoding='utf-8')
    wr = csv.writer(f_csv)
    # The first column is 'index', which we increment for every experiment
    wr.writerow(["index", "rank", "K", "gap",
                "v1_est", "v2_exact", "abs_error"])

    # We'll use a single 'index_count' across all distribution branches
    index_count = 0

    # 5) Conduct experiments
    if distribution == "random":
        # We assume iteration > 0, gaps=[-1]
        if not config.ranks or not config.k_values:
            print("[main] ranks or k_values is empty. Cannot proceed with random.")
            f_csv.close()
            return

        r = config.ranks[0]
        K = config.k_values[0]

        idx_list = []
        v1_list = []
        v2_list = []

        for it in range(config.iteration):
            np.random.seed(it)
            eigs = gen_random(r)
            if eigs is None:
                continue

            sums = sum_with_error(eigs, r, error_mag=0.0)
            a = cal_coefficient(sums, r)
            v1, v2 = cal_power_trace(sums, eigs, a, K, r)
            abs_err = abs(v1 - v2)

            # Write CSV row with index_count
            wr.writerow([index_count, r, K, -1, v1, v2, abs_err])
            index_count += 1

            idx_list.append(it)
            v1_list.append(v1)
            v2_list.append(v2)

        f_csv.close()

        if len(idx_list) == 0:
            print("[main] No data for random distribution.")
            return

        # Plot (random)
        abs_errors = [abs(v1_list[i] - v2_list[i])
                      for i in range(len(v1_list))]

        plt.figure(figsize=(10, 6))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(idx_list, v1_list, label="v1 (estimated)", color='blue')
        ax1.plot(idx_list, v2_list, label="v2 (exact)",
                 color='orange', linestyle='--')
        ax1.set_title(
            f"Random distribution (rank={r}, K={K}, iteration={config.iteration})")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Tr(rho^(r+K))")
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(idx_list, abs_errors, 'g-')
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Absolute Error")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot_random.png"), dpi=150)
        plt.close()

        print(
            f"[main] Random distribution done. Results saved in '{save_dir}'.")

    else:
        # distribution = "arithmetic" or "geometric"
        ranks = config.ranks
        k_vals = config.k_values
        gaps = config.gaps  # multiple gap values

        # We'll produce a separate heatmap for each gap
        for gap in gaps:
            # Build 2D matrix for absolute errors: dimension=(len(ranks), len(k_vals))
            abs_err_map = np.zeros((len(ranks), len(k_vals))) * np.nan

            for i, r in enumerate(ranks):
                for j, K in enumerate(k_vals):
                    if distribution == "arithmetic":
                        eigs = gen_arithmetic(gap, r)
                    else:  # "geometric"
                        eigs = gen_geometric(gap, r)

                    if eigs is None:
                        # record an entry with NaN
                        wr.writerow(
                            [index_count, r, K, gap, "nan", "nan", "nan"])
                        index_count += 1
                        abs_err_map[i, j] = np.nan
                        continue

                    sums = sum_with_error(eigs, r, error_mag=0.0)
                    a = cal_coefficient(sums, r)
                    v1, v2 = cal_power_trace(sums, eigs, a, K, r)
                    abs_err = abs(v1 - v2)

                    wr.writerow([index_count, r, K, gap, v1, v2, abs_err])
                    index_count += 1

                    abs_err_map[i, j] = abs_err

            # Create heatmap for this particular gap
            # (if all are NaN, we skip)
            if not np.all(np.isnan(abs_err_map)):
                plt.figure(figsize=(8, 6))
                im = plt.imshow(abs_err_map, origin='upper', cmap='viridis')
                plt.colorbar(im, label="Absolute Error |v1 - v2|")

                plt.xticks(np.arange(len(k_vals)), [str(k) for k in k_vals])
                plt.yticks(np.arange(len(ranks)), [str(rr) for rr in ranks])
                plt.xlabel("K values")
                plt.ylabel("Rank (r)")

                plt.title(
                    f"{distribution.capitalize()} distribution\n gap={gap}")
                plt.savefig(os.path.join(
                    save_dir, f"heatmap_{distribution}_gap{gap}.png"), dpi=150)
                plt.close()

        f_csv.close()
        print(
            f"[main] {distribution.capitalize()} distribution done (multiple gaps). Results in '{save_dir}'.")


if __name__ == "__main__":
    main()
