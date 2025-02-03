import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from collections import defaultdict

from utils import gen_arithmetic, sum_with_error, cal_coefficient, cal_power_trace


def experiment_arithmetic_A(r_vals, k_vals, eps_vals, gap=0.25, trials=10):
    """
    For each r in r_vals, for each k in k_vals, for each eps in eps_vals,
    do 'trials' times. Each time we generate sums with noise 'eps'
    and compute the absolute error. Then we store the mean error for that (r,k,eps).
    """
    results = []  # list of dict rows
    index_count = 0

    for r in r_vals:
        eigs = gen_arithmetic(gap, r)
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


def experiment_arithmetic_B(r, k_vals, gap_vals, eps=1e-6, trials=1):
    """
    For a fixed rank r, fixed eps, multiple gap values,
    and multiple k's in k_vals.

    Returns a data list of dict rows:
      [ { 'gap':..., 'k':..., 'mean_error':... }, ... ]
    """
    results = []
    index_count = 0

    for k in k_vals:
        for gap in gap_vals:
            eigs = gen_arithmetic(gap, r)
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
    # 1) Define parameters
    r_vals = [4, 8]
    k_vals = [16, 64, 32, 128]
    eps_vals = [1e-8, 1e-6, 1e-4]
    gap = 0.25
    trials = 1  # or more if needed

    # 2) Run experiment
    data = experiment_arithmetic_A(
        r_vals, k_vals, eps_vals, gap=gap, trials=trials)

    # 3) Write CSV
    with open("result/arith_A.csv", "w", newline='') as f:
        fieldnames = ["index", "r", "K", "eps", "mean_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    # 4) Read data into a dict: (r,k) -> list of (eps, mean_err)
    data_map = defaultdict(list)
    for row in data:
        r_ = row["r"]
        K_ = row["K"]
        eps_ = row["eps"]
        err_ = row["mean_error"]
        data_map[(r_, K_)].append((eps_, err_))

    # ---------------------------
    # 5) Define style mapping
    # ---------------------------
    style_map = {
        # r=4 => solid ('-') + circle ('o')
        # r=8 => dashed ('--') + star ('*')
        # k=16(red),64(blue),32(green),128(purple)
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

    # sort eps for each line
    for (r_, K_) in data_map:
        arr = sorted(data_map[(r_, K_)], key=lambda x: x[0])  # sort by eps
        xs = [p[0] for p in arr]
        ys = [p[1] for p in arr]

        # Lookup style
        plot_style = style_map.get((r_, K_), dict(
            linestyle='-', marker='o', color='black'))
        label_str = f"(r={r_}, K={K_})"

        # plot with style
        plt.plot(xs, ys, label=label_str, **plot_style)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("eps")
    plt.ylabel("Mean Absolute Error")
    plt.title("Arithmetic distribution: mean error vs eps")
    plt.legend()
    plt.grid(True)
    plt.savefig("result/arith_A.png", dpi=150)
    plt.show()


def main_B():
    # 1) Parameters
    r = 4
    k_vals = [16, 32, 64, 128]
    gap_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    eps = 1e-6
    trials = 1  # or more if you want averages

    # 2) Run experiment
    data = experiment_arithmetic_B(
        r, k_vals, gap_vals, eps=eps, trials=trials)

    # 3) Save to CSV (optional)
    csv_path = "result/arith_B.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["index", "r", "k", "gap", "eps", "mean_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    # 4) Group data by k => list of (gap, mean_err)
    from collections import defaultdict
    data_map = defaultdict(list)
    for row in data:
        k_ = row["k"]
        gap_ = row["gap"]
        err_ = row["mean_error"]
        data_map[k_].append((gap_, err_))

    # 5) Plot
    plt.figure(figsize=(7, 5))

    # For clarity, define a style map: k -> line style
    style_map = {
        16: dict(color='red', marker='o'),
        32: dict(color='green', marker='*'),
        64: dict(color='blue', marker='s'),
        128: dict(color='purple', marker='D'),
    }

    for k_ in k_vals:
        arr = sorted(data_map[k_], key=lambda x: x[0])  # sort by gap
        xs = [p[0] for p in arr]   # gap
        ys = [p[1] for p in arr]   # mean error

        plot_style = style_map.get(k_, dict(color='black', marker='o'))
        label_str = f"k={k_}, eps={eps}"

        plt.plot(xs, ys, label=label_str, linestyle='-', **plot_style)

    plt.yscale('log')
    plt.xlabel(r"$\lambda_{\max} - \lambda_{\min}$ (gap)")
    plt.ylabel("Absolute Error")
    plt.title(f"Arithmetic Dist (r={r}, eps={eps}): Error vs gap")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("result/arith_B.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main_A()
    main_B()
