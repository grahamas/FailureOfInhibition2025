# Optimized Parameters

This document explains how the oscillation optimizer saves and loads optimized parameters.

## Overview

The oscillation optimizer script (`scripts/optimize_oscillation_parameters.jl`) performs parameter space exploration to find Wilson-Cowan model parameters that produce sustained, robust oscillations. The results are automatically saved to a JSON file for reuse.

## How It Works

### 1. Running the Optimization

To run the parameter optimization:

```bash
julia --project=. scripts/optimize_oscillation_parameters.jl
```

The script will:
- Test baseline parameters from Wilson & Cowan 1973
- Explore different connectivity strengths
- Vary time constant ratios
- Test different nonlinearity parameters
- Save the best configuration found

### 2. Parameter Storage

Optimization results are saved to `data/optimized_parameters.json` with the following structure:

```json
{
    "mode": "oscillatory_optimized",
    "timestamp": "2025-11-11T21:46:19.736",
    "configuration": "τₑ=8.0, τᵢ=10.0",
    "metrics": {
        "score": 0.77,
        "has_oscillations": true,
        "n_peaks": 8,
        "decay_rate": 0.004152,
        "amplitude": 0.0059,
        "frequency": 0.029350,
        "half_life": 166.96,
        "period": 34.07
    },
    "parameters": {
        "tau_e": 8.0,
        "tau_i": 10.0,
        "alpha_e": 1.0,
        "alpha_i": 1.0,
        "beta_e": 1.0,
        "beta_i": 1.0,
        "connectivity": {
            "b_ee": 2.2,
            "b_ei": -1.5,
            "b_ie": 1.5,
            "b_ii": -0.08
        },
        "nonlinearity": {
            "v_e": 0.5,
            "theta_e": 9.0,
            "v_i": 1.0,
            "theta_i": 15.0
        }
    }
}
```

### 3. Using Optimized Parameters

The optimized parameters are automatically loaded when you create a model with the `:oscillatory_optimized` mode:

```julia
using FailureOfInhibition2025
include("test/test_wcm1973_validation.jl")

# Create a point model with optimized parameters
params = create_point_model_wcm1973(:oscillatory_optimized)

# Use the parameters for simulation
A₀ = reshape([0.3, 0.2], 1, 2)
tspan = (0.0, 300.0)
sol = solve_model(A₀, tspan, params, saveat=0.5)
```

### 4. Required File

**Important**: The JSON file must exist before using `:oscillatory_optimized` mode. If the file doesn't exist, the function will error with a helpful message:

```
ERROR: Optimized parameters file not found at: /path/to/data/optimized_parameters.json
Run the optimization script first: julia --project=. scripts/optimize_oscillation_parameters.jl
```

This ensures that you always use properly optimized parameters rather than potentially outdated hardcoded values.

## Updating Parameters

To update the optimized parameters:

1. Modify the optimization script if needed (e.g., change parameter ranges, scoring function)
2. Run the optimization script: `julia --project=. scripts/optimize_oscillation_parameters.jl`
3. The new parameters will be saved to `data/optimized_parameters.json`
4. All code using `:oscillatory_optimized` mode will automatically use the new parameters

## Comparison with Baseline

You can compare the optimized parameters with the baseline Wilson & Cowan 1973 parameters:

```julia
# Baseline WCM 1973 oscillatory mode
params_baseline = create_point_model_wcm1973(:oscillatory)

# Optimized parameters
params_optimized = create_point_model_wcm1973(:oscillatory_optimized)
```

See `scripts/plot_optimized_oscillations.jl` for a visual comparison showing the improvement in oscillation amplitude and robustness.

## Benefits

- **Reproducibility**: Optimization results are stored and can be reused
- **Easy Updates**: Re-run optimization to find better parameters
- **No Manual Editing**: Parameters update automatically without code changes
- **Metadata Tracking**: Timestamps and metrics help track optimization history
- **Explicit Dependencies**: Errors if optimization hasn't been run, ensuring current parameters are used
