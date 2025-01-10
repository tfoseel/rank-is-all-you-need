# Rank Is All You Need: Estimating the Trace of Powers of Density Matrices

This repository stores the code for our work [Rank Is All You Need: Estimating the Trace of Powers of Density Matrices](https://arxiv.org/abs/2408.00314).

The code is separated into three files:

1. ```config.py```  
   - Contains configuration variables: `distribution`, `ranks`, `k_values`, `iteration`, and `gaps`.
   - Depending on the distribution, certain parameters (like `gap` or `iteration`) are either used or ignored.

2. ```utils.py```
   - Contains utility functions for generating eigenvalues (`random`, `arithmetic`, `geometric`) and computing Newton-Girard coefficients (`sum_with_error`, `cal_coefficient`, `cal_power_trace`).
   - These functions focus on the mathematical logic for generating eigenvalue vectors and calculating $\mathrm{Tr}(\rho^k)$.

3. ```main.py```  
   - Reads the configuration from `config.py` and runs the experiment.
   - Saves the results (CSV and plots) into a timestamped folder under `result/`.
   - Also copies the current `config.py` into that folder for reference.

## Installation

1. **Install Python 3** if you don't have it already.
2. Install the required Python libraries:
   - `numpy`
   - `matplotlib`
   - You can install it by ```pip install numpy matplotlib```.

## How to Run

1. Place `config.py`, `utils.py`, and `main.py` in the same directory.
2. Edit `config.py` to set:
- `distribution` in `["random", "arithmetic", "geometric"]`
- For **random**:
  - `iteration` > 0 (number of times to generate random eigenvalues)
  - `gaps = [-1]` (ignored)
- For **arithmetic** or **geometric**:
  - `iteration = -1`
  - `gaps = [ ... ]` (list of gap or ratio values)
  - `ranks = [ ... ]`
  - `k_values = [ ... ]`
3. Run the experiment via the following command:
   - ```python -m main```.
4. The script will create a `result/` folder (if missing), then a subfolder named by the current timestamp (e.g. `20250108_123456`).
5. Inside that subfolder, you will find:
   - `config.py` (a copy of the config used at runtime, and can be used for reproducing the result)
   - `results.csv` (the main experiment output)
   - Plot files (`plot_random.png`, `heatmap_arithmetic_gap0.5.png`, etc.)

## Output Details

### ```results.csv```  

This file stores the numerical simulation results using the following columns:
 1. `index` (sequential integer for each experiment)  
 2. `rank`  
 3. `K`  
 4. `gap`  
 5. `v1_est` (estimated $\mathrm{Tr}(\rho^K)$)  
 6. `v2_exact` (exact $\mathrm{Tr}(\rho^K)$)  
 7. `abs_error` (`|v1 - v2|`)  

### Plots  

For **random** distribution, two subplots are stored, where each subplots are:
 1. Iteration vs. `(v1, v2)`
 2. Iteration vs. `abs_error`

For **arithmetic** or **geometric**, a heatmap of `abs_error` over `(rank, K)` for each gap is stored.

You can create other plots using the ```results.csv``` file.

## Extending Functionality

### 1. Adding a New Eigenvalue Generator

You can define a **new** distribution in `utils.py`. For example:

```python
def gen_mynewdist(param1, size): 
    # Use numpy or other logic to create an eigenvalue vector of length 'size'
    # Must be non-negative and sum to 1.
    # If something goes wrong, return None.
    # Otherwise, return an array or list of eigenvalues.
    pass
```

Then, in `main.py` (or wherever distribution is chosen), add:

```python
if distribution == "mynewdist":
    eigs = gen_mynewdist(your_param, r)
else: ...
```

You can store extra parameters in `config.py` (similar to `gaps` or `iteration`). The `sum_with_error`, `cal_coefficient`, and `cal_power_trace` functions should remain unchanged, as they only depend on the generated eigenvalues.

### 2. Modifying the Workflow

- If you want to handle additional logic (e.g., multiple parameters for a new distribution), expand `config.py` with more variables and adapt `main.py` accordingly.
- If you want a different output format, edit how you write rows in `results.csv` or how you produce your plots.