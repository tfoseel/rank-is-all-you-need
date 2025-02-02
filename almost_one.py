import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from collections import defaultdict

# 가정: utils.py 등에 이미 gen_almost_one(r)가 정의되어 있음
from utils import gen_almost_one, sum_with_error, cal_coefficient, cal_power_trace


def experiment_almost_one(r_vals, k_vals, eps_vals, trials=10):
    """
    For each r in r_vals, for each k in k_vals, for each eps in eps_vals,
    we do 'trials' runs.
    Each run:
      - Generate an eigenvalue vector via gen_almost_one(r)
      - Add measurement noise eps (sum_with_error)
      - Compare Tr(rho^k) estimated vs exact
    Then compute the mean error for that (r,k,eps).
    """
    results = []
    index_count = 0

    for r in r_vals:
        # 1) Generate the "almost-one" eigenvalues of length r
        # largest eigenvalue in [0.999..0.999999], rest random
        eigs = gen_almost_one(r)
        if eigs is None or abs(sum(eigs) - 1.0) > 1e-8:
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


def main():
    # 1) Parameters
    r_vals = [4, 8]
    k_vals = [16, 64, 32, 128]
    eps_vals = [1e-8, 1e-6, 1e-4]
    trials = 1  # or more as needed

    # 2) Run experiment with gen_almost_one
    data = experiment_almost_one(r_vals, k_vals, eps_vals, trials=trials)

    # 3) Write CSV
    with open("result/almost_one.csv", "w", newline='') as f:
        fieldnames = ["index", "r", "K", "eps", "mean_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    # 4) Group data by (r,k)
    data_map = defaultdict(list)
    for row in data:
        r_ = row["r"]
        K_ = row["K"]
        eps_ = row["eps"]
        err_ = row["mean_error"]
        data_map[(r_, K_)].append((eps_, err_))

    # 5) Define style mapping (same as before)
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
        xs = [p[0] for p in arr]
        ys = [p[1] for p in arr]

        plot_style = style_map.get((r_, K_), dict(
            linestyle='-', marker='o', color='black'))
        label_str = f"(r={r_}, K={K_})"

        plt.plot(xs, ys, label=label_str, **plot_style)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("eps")
    plt.ylabel("Mean Absolute Error")
    plt.title("gen_almost_one: mean error vs eps")
    plt.legend()
    plt.grid(True)
    plt.savefig("result/almost_one.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
