import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from collections import defaultdict

from utils import gen_geometric, sum_with_error, cal_coefficient, cal_power_trace


def experiment_geometric_A(r_vals, k_vals, eps_vals, gap=0.25, trials=10):
    """
    For each r in r_vals, for each k in k_vals, for each eps in eps_vals:
      - Generate eigenvalues via gen_geometric(gap, r).
      - 'trials' times add noise eps, compute absolute error, then average it.
    Returns a list of dict rows with (r, k, eps, mean_error).
    """
    results = []
    index_count = 0

    for r in r_vals:
        eigs = gen_geometric(gap, r)
        if eigs is None:
            continue

        for k in k_vals:
            for eps in eps_vals:
                abs_errs = []
                for _ in range(trials):
                    tilde_r = r
                    # tilde_r = math.ceil(math.log(k / eps))
                    # tilde_r = math.ceil(math.log(k / eps) / math.log(math.log(k / eps)))
                    sums = sum_with_error(eigs, tilde_r, eps)
                    a = cal_coefficient(sums, tilde_r)
                    v1, v2 = cal_power_trace(sums, eigs, a, k, tilde_r)
                    abs_err = abs(v1 - v2)
                    abs_errs.append(abs_err)

                mean_err = np.mean(abs_errs)
                row = {
                    "index": index_count,
                    "r": r,
                    "K": k,
                    "eps": eps,
                    "mean_error": mean_err
                }
                results.append(row)
                index_count += 1

    return results


def experiment_geometric_B(r, k_vals, gap_vals, eps=1e-6, trials=1):
    """
    For a fixed rank r, fixed eps, multiple 'gap' values (geometric ratio),
    and multiple k in k_vals.
    We interpret 'gap' as max_min_ratio for gen_geometric.
    """
    results = []
    index_count = 0

    for k in k_vals:
        for gap in gap_vals:
            eigs = gen_geometric(gap, r)
            if eigs is None:
                continue

            abs_errs = []
            for _ in range(trials):
                tilde_r = r
                # tilde_r = math.ceil(math.log(k / eps))
                # tilde_r = math.ceil(math.log(k / eps) / math.log(math.log(k / eps)))
                sums = sum_with_error(eigs, tilde_r, eps)
                a = cal_coefficient(sums, tilde_r)
                v1, v2 = cal_power_trace(sums, eigs, a, k, tilde_r)
                abs_err = abs(v1 - v2)
                abs_errs.append(abs_err)

            mean_err = np.mean(abs_errs)
            row = {
                "index": index_count,
                "r": r,
                "k": k,
                "gap": gap,
                "eps": eps,
                "mean_error": mean_err
            }
            results.append(row)
            index_count += 1

    return results


def main_A():
    # 1) Define parameters for the "A" scenario
    r_vals = [4, 8]
    k_vals = [16, 32, 64, 128]
    eps_vals = [1e-8, 1e-6, 1e-4]
    gap = 1024
    trials = 1

    # 2) Run experiment
    data = experiment_geometric_A(
        r_vals, k_vals, eps_vals, gap=gap, trials=trials)

    # 3) Write CSV => "geom_A.csv"
    csv_file = "result/geom_A.csv"
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ["index", "r", "K", "eps", "mean_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    # 4) Group data (r,k) -> list of (eps, mean_err)
    data_map = defaultdict(list)
    for row in data:
        r_ = row["r"]
        K_ = row["K"]
        eps_ = row["eps"]
        err_ = row["mean_error"]
        data_map[(r_, K_)].append((eps_, err_))

    # 5) Style map (like before)
    style_map = {
        (4, 16): dict(linestyle='-', marker='o', color='red'),
        (4, 32): dict(linestyle='-', marker='o', color='green'),
        (4, 64): dict(linestyle='-', marker='o', color='blue'),
        (4, 128): dict(linestyle='-', marker='o', color='purple'),

        (8, 16): dict(linestyle='--', marker='*', color='red'),
        (8, 32): dict(linestyle='--', marker='*', color='green'),
        (8, 64): dict(linestyle='--', marker='*', color='blue'),
        (8, 128): dict(linestyle='--', marker='*', color='purple'),
    }

    # 6) Plot
    plt.figure(figsize=(8, 6))
    for (r_, K_) in data_map:
        arr = sorted(data_map[(r_, K_)], key=lambda x: x[0])  # sort by eps
        xs = [p[0] for p in arr]  # eps
        ys = [p[1] for p in arr]  # mean_error

        plot_style = style_map.get((r_, K_), dict(
            linestyle='-', marker='o', color='black'))
        label_str = f"(r={r_}, K={K_})"
        plt.plot(xs, ys, label=label_str, **plot_style)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("eps")
    plt.ylabel("Mean Absolute Error")
    plt.title("Geometric Dist (Exp A): mean error vs eps")
    plt.legend()
    plt.grid(True)
    plt.savefig("result/geom_A.png", dpi=150)
    plt.show()


def main_B():
    # 1) Parameters for the "B" scenario
    r = 4
    k_vals = [16, 32, 64, 128]
    # For geometric distribution, 'gap' is actually the max_min_ratio
    gap_vals = [2, 4, 8, 16, 32, 64, 128]
    eps = 1e-6
    trials = 1

    # 2) Run the geometric experiment B
    data = experiment_geometric_B(r, k_vals, gap_vals, eps=eps, trials=trials)

    # 3) CSV => "geom_B.csv"
    csv_path = "result/geom_B.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["index", "r", "k", "gap", "eps", "mean_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    # 4) Group by k => list of (gap, mean_err)
    data_map = defaultdict(list)
    for row in data:
        k_ = row["k"]
        gap_ = row["gap"]
        err_ = row["mean_error"]
        data_map[k_].append((gap_, err_))

    # 5) style map for k
    style_map = {
        16: dict(color='red', marker='o'),
        32: dict(color='green', marker='*'),
        64: dict(color='blue', marker='s'),
        128: dict(color='purple', marker='D'),
    }

    # 6) Plot
    plt.figure(figsize=(7, 5))
    for k_ in k_vals:
        arr = sorted(data_map[k_], key=lambda x: x[0])  # sort by gap
        xs = [p[0] for p in arr]
        ys = [p[1] for p in arr]

        plot_style = style_map.get(k_, dict(color='black', marker='o'))
        label_str = f"k={k_}, eps={eps}"
        plt.plot(xs, ys, label=label_str, linestyle='-', **plot_style)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"max_min_ratio for geometric")
    plt.ylabel("Mean Absolute Error")
    plt.title(f"Geometric Dist (Exp B, r={r}, eps={eps}): Error vs ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("result/geom_B.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main_A()
    main_B()
