# Bifurcation Diagrams for Full Dynamics Models

This directory contains bifurcation analysis results for the `full_dynamics_monotonic` and `full_dynamics_blocking` models.

## Overview

Bifurcation diagrams show how the steady states of a dynamical system change as a parameter varies. These analyses help understand:
- Critical parameter values where system behavior changes qualitatively
- Stability of steady states (stable vs. unstable)
- Existence of multiple steady states (bistability, multistability)
- Transitions between different dynamical regimes

## Models Analyzed

### Full Dynamics Monotonic
Point model version of the full dynamics model with monotonic (standard sigmoid) inhibition.
- E population: SigmoidNonlinearity (a=50.0, θ=0.125)
- I population: SigmoidNonlinearity (a=50.0, θ=0.2)
- Decay rates: αE=0.4, αI=0.7
- Time constants: τE=1.0, τI=0.4

### Full Dynamics Blocking  
Point model version of the full dynamics model with blocking (non-monotonic) inhibition.
- E population: RectifiedZeroedSigmoidNonlinearity (a=50.0, θ=0.125)
- I population: DifferenceOfSigmoidsNonlinearity (captures failure of inhibition)
  - Activating sigmoid: a=50.0, θ=0.2
  - Blocking sigmoid: a=50.0, θ=0.5
- Decay rates: αE=0.4, αI=0.7
- Time constants: τE=1.0, τI=0.4

## Analysis Results

### Stimulus Strength Effect ✓ COMPLETED

**Files:**
- `monotonic_stimulus_strength.csv` - Numerical data for monotonic model
- `monotonic_stimulus_strength.png` - Bifurcation diagram for monotonic model
- `blocking_stimulus_strength.csv` - Numerical data for blocking model
- `blocking_stimulus_strength.png` - Bifurcation diagram for blocking model

**Parameter Range:** Stimulus strength varied from 0.0 to 5.0

**Key Findings:**
- Both models show how constant external stimulus affects steady state activity levels
- Bifurcation diagrams reveal stable and unstable branches
- Can identify critical stimulus strengths where qualitative changes occur
- Differences between monotonic and blocking inhibition are visible in the bifurcation structure

## How to Regenerate

To regenerate these bifurcation diagrams, run:

```bash
julia --project=. scripts/generate_bifurcation_diagrams_full_dynamics.jl
```

The script will:
1. Create point model versions of the spatial full dynamics models
2. Run continuation analysis using BifurcationKit
3. Save CSV data files with parameter values and steady states
4. Generate PNG plots of the bifurcation diagrams
5. Save all results to this directory

## Data Format

CSV files contain columns:
- `parameter` - The parameter value being varied
- `var_1`, `var_2`, ... - Steady state values for each variable (E and I populations)
- `stable` (if available) - Boolean indicating stability of the branch

## Notes

- Point models are used for bifurcation analysis (non-spatial)
- Connectivity values are extracted from the spatial models' Gaussian parameters
- Continuation uses the PALC (Pseudo-Arc-Length Continuation) method
- Initial conditions are found by solving the system to steady state

## Future Work

Additional parameter analyses could include:
- E→E connectivity strength
- I→E connectivity strength (inhibition)
- Nonlinearity threshold parameters
- Two-parameter bifurcation diagrams

These require careful tuning of continuation parameters and initial conditions to avoid numerical issues.

## References

- Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory of the functional dynamics of cortical and thalamic nervous tissue. *Kybernetik*, 13(2), 55-80.
- Harris, K. D., & Ermentrout, G. B. (2018). Bifurcations in the Wilson-Cowan equations with distributed delays. *SIAM Journal on Applied Dynamical Systems*, 17(1), 501-520.
- BifurcationKit.jl documentation: https://bifurcationkit.github.io/BifurcationKitDocs.jl/
