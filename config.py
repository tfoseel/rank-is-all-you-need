#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration file for the Newton-Girard trace estimation experiments.

- distribution: can be "random", "arithmetic", or "geometric".
  * If distribution == "random":
    - iteration is used to run multiple experiments on random eigenvalues.
    - gaps is set to [-1] (unused).
  * If distribution == "arithmetic" or "geometric":
    - iteration is set to -1 (unused).
    - gaps is a list of gap values (e.g. [0.0, 0.25, 0.5]) or ratio values for geometric.

- ranks: a list of integer ranks.
- k_values: a list of integer K values.
"""

distribution = "arithmetic"   # "random", "arithmetic", or "geometric"

if distribution == "random":
    iteration = 1000   # Used only in 'random' case
    gaps = [-1]        # Not used in random distribution
else:
    iteration = -1     # Not used in arithmetic/geometric
    # You can put multiple gap values here
    gaps = [0.0, 0.25, 0.5]

ranks = [4, 8]
k_values = [16, 32, 64, 128]
