#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for eigenvalue generation and Newton-Girard trace estimation.
"""

import numpy as np


def sum_with_error(eigs, r, error_mag):
    """
    Compute sums[i] = Tr(rho^i) + noise in [-error_mag, +error_mag].
    sums[0] = 1 as a convention for Newton-Girard usage.
    """
    sums = [1.0]
    for i in range(1, r + 1):
        true_val = sum(lam**i for lam in eigs)
        error = 2 * error_mag * np.random.rand() - error_mag
        sums.append(true_val + error)
    return sums


def cal_coefficient(sums, r):
    """
    Calculate Newton-Girard coefficients a[0..r].
    a[0] = 1, a[1] = sums[1], etc.
    """
    a = [1.0, sums[1]]
    for i in range(2, r + 1):
        v = 0.0
        for j in range(1, i + 1):
            v += ((-1)**(j-1)) * a[i - j] * sums[j]
        a.append(v / i)
    return a


def cal_power_trace(sums, eigen_values, a, K, r):
    """
    Use Newton-Girard to expand sums up to r+K.
    Return v1 (Tr(rho^(r+K)) estimated) and v2 (exact).
    """
    v1 = 0.0
    for i in range(1, K + 1):
        tmp = 0.0
        for j in range(1, r + 1):
            tmp += ((-1)**(j-1)) * sums[r - j + i] * a[j]
        v1 = tmp
        sums.append(tmp)
    v2 = sum(lam**(r + K) for lam in eigen_values)
    return v1, v2


def gen_arithmetic(max_min_gap, size):
    """
    Generate an eigenvalue vector in arithmetic progression (size = 'size'),
    where (max_value - min_value) = max_min_gap.
    If it's not possible to construct such a distribution, return None.

    Example:
    --------
    gen_arithmetic(0, 4) -> [0.25, 0.25, 0.25, 0.25]
    gen_arithmetic(0.5, 4) -> might look like [0.5, 0.3333, 0.1667, 0.0], scaled to sum=1
    """
    if max_min_gap > 2 / size or max_min_gap > 2 - 2 / size:
        print("[gen_arithmetic] Error: Cannot satisfy the given max_min_gap.")
        return None
    eigs = np.linspace(1/size + max_min_gap/2, 1/size - max_min_gap/2, size)
    s = np.sum(eigs)
    if s == 0:
        return None
    eigs /= s
    return eigs


def gen_geometric(max_min_ratio, size):
    """
    Generate an eigenvalue vector in geometric progression (size = 'size'),
    where (max_value / min_value) = max_min_ratio.
    If it's not possible, return None.

    Example:
    --------
    gen_geometric(2, 4) -> might look like [0.4, 0.2667, 0.2, 0.1333], scaled to sum=1
    """
    if max_min_ratio <= 1:
        print("[gen_geometric] Error: max_min_ratio must be > 1.")
        return None
    r = max_min_ratio ** (1 / (1 - size))
    numerator = (1 - r)
    denominator = (1 - r**size)
    if denominator == 0:
        return None
    a_1 = numerator / denominator
    a_size = numerator * (r**(size - 1)) / denominator
    eigs = np.geomspace(a_1, a_size, num=size)
    s = np.sum(eigs)
    if s == 0:
        return None
    eigs /= s
    return eigs


def gen_almost_one(size):
    """
    Generate an eigenvalue distribution of length `size` where:
      - One eigenvalue is chosen in the range [0.999, 0.999999].
      - The rest (size-1) eigenvalues share 1 - that large value randomly.
    Returns:
      A NumPy array of shape (size,) that sums to 1.
    """
    if size < 2:
        raise ValueError("size must be at least 2 for gen_almost_one")
    big_val = np.random.uniform(0.999, 0.999999)
    remain = 1.0 - big_val
    other = np.random.rand(size - 1)
    other_sum = np.sum(other)
    if other_sum > 0:
        other = other / other_sum * remain
    else:
        other = np.zeros(size - 1)
    eigs = np.concatenate(([big_val], other))
    eigs = np.sort(eigs)[::-1]
    return eigs


def gen_random(size):
    """
    Generate a random eigenvalue vector (size = 'size'), summing to 1, all >= 0.
    This method splits [0,1] in random ratios to create eigenvalues.
    """
    diff_values = np.random.rand(size - 1)
    eigs = []
    v = 1.0
    for d in diff_values:
        eigs.append(v * d)
        v *= (1 - d)
    eigs.append(v)
    eigs = np.array(eigs)
    s = np.sum(eigs)
    if s == 0:
        return None
    eigs /= s
    return eigs
