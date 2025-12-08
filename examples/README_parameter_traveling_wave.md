# Parameter-Traveling Wave Relationship Analysis

## Overview

The `example_parameter_traveling_wave_relationship.jl` example provides a comprehensive analysis of how model parameters affect traveling wave properties in 1D Wilson-Cowan neural field models.

## What This Example Does

This example systematically explores the parameter space of a 1D Wilson-Cowan model to understand:

1. **How connectivity width affects wave propagation**
   - Narrow connectivity: waves fail to propagate
   - Moderate connectivity: optimal for sustained waves
   - Wide connectivity: may lead to stationary bumps

2. **How sigmoid steepness controls activation dynamics**
   - Low steepness: gradual activation, slower waves
   - Moderate steepness: balanced propagation
   - High steepness: faster but potentially unstable waves

3. **How sigmoid threshold affects wave initiation**
   - Low threshold: easy activation but quick saturation
   - Moderate threshold: balanced behavior
   - High threshold: waves may fail to initiate

4. **How time constant influences temporal dynamics**
   - Small τ: fast but potentially unsustained responses
   - Moderate τ: optimal balance
   - Large τ: slower, longer-lasting waves

## Generated Visualizations

Running this example produces 5 PNG files:

### 1. `param_sweep_connectivity_width.png`
Four subplots showing how connectivity width affects:
- Distance traveled
- Wave speed
- Decay rate
- Amplitude

### 2. `param_sweep_sigmoid_steepness.png`
Four subplots showing how sigmoid steepness (a parameter) affects wave properties.

### 3. `param_sweep_sigmoid_threshold.png`
Four subplots showing how sigmoid threshold (θ parameter) affects wave properties.

### 4. `param_sweep_time_constant.png`
Four subplots showing how time constant (τ parameter) affects wave properties.

### 5. `param_space_2d.png`
Two heatmaps showing the joint effects of:
- Connectivity width × Sigmoid steepness on distance traveled
- Connectivity width × Sigmoid steepness on wave speed

## Running the Example

```bash
cd /path/to/FailureOfInhibition2025
julia --project=. examples/example_parameter_traveling_wave_relationship.jl
```

The example takes approximately 2-3 minutes to run (depending on hardware) and produces both console output with numerical results and PNG visualizations.

## Key Findings

The analysis reveals several important relationships:

1. **Non-linear parameter interactions**: Parameters don't act independently - their effects combine in complex ways

2. **Optimal parameter regimes**: Multiple parameter combinations can produce similar behaviors, but certain combinations are optimal for specific goals

3. **Phase transitions**: Some parameter changes lead to qualitative shifts in behavior (e.g., traveling → stationary)

4. **Design principles**: For maximum wave propagation:
   - Use moderate connectivity width (2.5-3.5)
   - Set sigmoid steepness to 2.0-2.5
   - Use threshold around 0.20-0.25

## Applications

This analysis is particularly useful for:

- **Understanding failure of inhibition**: Identifying parameter regimes where waves succeed vs. fail
- **Model tuning**: Finding parameters that match experimental observations
- **Hypothesis generation**: Predicting how parameter changes affect dynamics
- **Drug/intervention studies**: Understanding how parameter perturbations affect wave propagation

## Related Examples

- `example_traveling_wave_behaviors.jl`: Demonstrates different types of traveling wave dynamics
- `example_optimize_traveling_waves.jl`: Shows how to automatically find optimal parameters
- `example_traveling_wave_metrics.jl`: Demonstrates individual traveling wave analysis metrics

## Citation

If you use this example in your research, please cite:

```
Graham Smith. (2025). FailureOfInhibition2025: 
A Julia package for neural field modeling with failure of inhibition mechanisms.
https://github.com/grahamas/FailureOfInhibition2025
```
