# Parameter Search for Oscillatory Behavior

## Overview

This document summarizes the parameter search conducted to find Wilson-Cowan model parameters that produce sustained oscillatory behavior. The search was motivated by the observation that the "oscillatory mode" parameters from Wilson & Cowan 1973 (Table 2) produce only **damped oscillations** rather than sustained oscillations.

## Background

The Wilson-Cowan model describes the dynamics of excitatory (E) and inhibitory (I) neural populations:

```
τₑ dE/dt = -αₑ E + βₑ (1 - E) σₑ(Input_E)
τᵢ dI/dt = -αᵢ I + βᵢ (1 - I) σᵢ(Input_I)

where:
Input_E = bₑₑ * E - bᵢₑ * I + Stimulus
Input_I = bₑᵢ * E - bᵢᵢ * I
```

For sustained oscillations to occur, the system needs:
1. **Strong E-I feedback loop**: Excitation drives inhibition, which suppresses excitation
2. **Phase lag**: Typically achieved through differential time constants (τᵢ > τₑ)
3. **Appropriate gain**: Sigmoid parameters that allow sufficient response
4. **Stability**: Parameters must avoid runaway excitation or complete suppression

## Wilson & Cowan 1973 "Oscillatory Mode" Parameters

The 1973 paper provides parameters for an "oscillatory mode" (Table 2):

| Parameter | Value | Description |
|-----------|-------|-------------|
| vₑ | 0.5 | Excitatory sigmoid steepness |
| θₑ | 9.0 | Excitatory threshold |
| vᵢ | 1.0 | Inhibitory sigmoid steepness (steeper than active transient) |
| θᵢ | 15.0 | Inhibitory threshold |
| bₑₑ | 2.0 | E → E coupling (stronger than active transient) |
| bᵢₑ | 1.5 | I → E coupling |
| bₑᵢ | 1.5 | E → I coupling |
| bᵢᵢ | 0.1 | I → I coupling (much weaker than active transient) |
| τ | 10.0 msec | Time constant (same for both populations) |

**Result**: These parameters produce **damped oscillations**, not sustained ones.

## Parameter Search Results

### Search Strategy 1: Connectivity Variations

**Approach**: Systematically varied connectivity strengths (bₑₑ, bᵢₑ, bₑᵢ, bᵢᵢ) around baseline values.

**Tested ranges**:
- bₑₑ: 2.0 to 8.0 (strengthening E-E coupling)
- bᵢₑ: 1.5 to 6.0 (strengthening I-E coupling) 
- bᵢᵢ: 0.0 to 0.1 (weakening I-I coupling)

**Result**: No sustained oscillations found. Strong coupling (bₑₑ > 3.0, bᵢₑ > 2.5) caused numerical instability.

### Search Strategy 2: Sigmoid Parameter Variations

**Approach**: Varied sigmoid slopes and thresholds to increase system gain.

**Tested ranges**:
- vᵢ: 1.0 to 2.5 (steeper inhibitory response)
- θₑ: 4.0 to 9.0 (lower excitatory threshold)
- θᵢ: 8.0 to 15.0 (lower inhibitory threshold)

**Result**: Lower thresholds (θₑ = 5.0-6.0, θᵢ = 10.0-12.0) produced **transient oscillations** (4 peaks detected in late window), but oscillations still decayed. Amplitude remained very small (< 0.001).

### Search Strategy 3: Differential Time Constants

**Approach**: Introduced different time constants for E and I populations to create phase lag.

**Theory**: τᵢ > τₑ creates delayed inhibitory response, which is necessary for sustained oscillations in many E-I network models.

**Tested**:
- τₑ: 5.0 to 10.0 msec
- τᵢ: 15.0 to 25.0 msec
- Ratios τᵢ/τₑ: 1.5 to 3.0

**Result**: No sustained oscillations. Differential time constants alone insufficient with WCM 1973 parameters.

### Search Strategy 4: Combined Optimizations

**Approach**: Tested combinations of:
- Differential time constants
- Stronger E-I feedback
- Lower thresholds

**Representative combinations tested**:
1. τₑ=8.0, τᵢ=20.0, bₑₑ=3.5, bᵢₑ=3.0, bₑᵢ=3.0
2. τₑ=7.0, τᵢ=18.0, bₑₑ=4.5, bᵢₑ=3.5, bₑᵢ=3.5
3. τₑ=8.0, τᵢ=22.0, bₑₑ=3.0, θₑ=6.0, θᵢ=12.0

**Result**: Strong coupling parameters caused numerical instability (blow-up) within 50 msec of simulation.

## Key Findings

### 1. The "Oscillatory Mode" from WCM 1973 is Damped

The parameters from Table 2 of Wilson & Cowan 1973 produce damped oscillations, not sustained ones. This is likely intentional in the paper, as they were demonstrating qualitative behaviors rather than claiming sustained oscillations.

### 2. Parameter Constraints Create a Trilemma

The Wilson-Cowan point model (non-spatial) with these parameter ranges exhibits three possible behaviors:

- **Too weak coupling** → Damped dynamics (system returns to rest)
- **Moderate coupling** → Transiently oscillatory but damped
- **Too strong coupling** → Numerical instability (blow-up)

There is a very narrow (possibly non-existent) parameter regime between "damped" and "unstable" that produces sustained oscillations in the point model.

### 3. Transient Oscillations Can Be Enhanced

Lower sigmoid thresholds (θₑ ≈ 5-6, θᵢ ≈ 10-12) combined with differential time constants (τᵢ > τₑ) produce more prominent **transient oscillations** with 4+ cycles. However, these still decay to rest.

### 4. Baseline-Subtracted Sigmoids Cause Instability

Using `RectifiedZeroedSigmoidNonlinearity` (which subtracts baseline like in the 1973 paper) caused numerical issues (NaN values) with lower time constants (τ < 5.0), making parameter exploration difficult.

## Recommendations

### For Sustained Oscillations in Wilson-Cowan Models

Based on the parameter search and theoretical considerations, sustained oscillations in Wilson-Cowan models typically require:

#### Option 1: Spatial Models

**Recommendation**: Use spatially extended models with wave propagation.

Spatial connectivity allows traveling waves and standing wave patterns that can sustain oscillations even when the local point dynamics would be damped. The 1973 paper demonstrates this with edge enhancement and sustained spatial patterns.

```julia
# Use create_wcm1973_parameters instead of create_point_model_wcm1973
params = create_wcm1973_parameters(:oscillatory)
# This includes spatial lattice and Gaussian connectivity kernels
```

#### Option 2: External Forcing

**Recommendation**: Apply periodic or sustained external stimulation.

```julia
# Sustained stimulus during analysis window
function sustained_stimulus(t; strength=10.0)
    return strength  # Constant input
end

# Or periodic stimulus
function periodic_stimulus(t; period=20.0, strength=10.0)
    return strength * (1 + sin(2π * t / period)) / 2
end
```

#### Option 3: Modified Dynamics

If sustained oscillations in a point model are required, consider:

1. **Different model**: Use FitzHugh-Nagumo, Hindmarsh-Rose, or other models explicitly designed for sustained oscillations

2. **Modified Wilson-Cowan**: Add:
   - Adaptation currents
   - Synaptic dynamics (slower than membrane dynamics)
   - Network heterogeneity

3. **Alternative formulations**: Use threshold-linear or piecewise-linear nonlinearities instead of sigmoids

### For Enhanced Transient Oscillations

If transient oscillations are acceptable:

```julia
params = create_ei_oscillatory_params(
    vₑ = 0.5,    # Standard excitatory sigmoid steepness
    θₑ = 6.0,    # Lower threshold (easier to activate)
    vᵢ = 1.0,    # Standard inhibitory sigmoid steepness  
    θᵢ = 12.0,   # Lower threshold (easier to activate)
    bₑₑ = 2.0,   # Moderate E-E coupling
    bᵢₑ = 1.5,   # Moderate I-E coupling
    bₑᵢ = 1.5,   # Moderate E-I coupling
    bᵢᵢ = 0.1,   # Weak I-I coupling
    τₑ = 8.0,    # Faster excitatory dynamics
    τᵢ = 18.0    # Slower inhibitory dynamics (creates phase lag)
)
```

This produces 4+ cycles of transient oscillations with a period of approximately 30-40 msec.

## Theoretical Considerations

### Why Sustained Oscillations Are Rare in Point WCM

1. **Sigmoid saturation**: The sigmoid nonlinearity bounds activity, preventing runaway excitation but also limiting oscillation amplitude

2. **No intrinsic time delays**: Unlike Hodgkin-Huxley models with voltage-gated channels, WCM has no intrinsic delays beyond membrane time constants

3. **Stable fixed points**: The WCM equations typically have stable fixed points that attract dynamics

4. **Lack of negative conductances**: Unlike some neural models, WCM doesn't have mechanisms for negative resistance that can sustain oscillations

### Conditions for Sustained Oscillations (Hopf Bifurcation)

Sustained oscillations require a **Hopf bifurcation** where a stable fixed point becomes unstable and a limit cycle emerges. This requires:

1. **Complex eigenvalues**: The linearization around the fixed point must have complex eigenvalues with positive real parts

2. **Delayed negative feedback**: Inhibition must lag excitation sufficiently (τᵢ >> τₑ or synaptic delays)

3. **Proper coupling strength**: Must be in narrow regime between damped and unstable

The parameter search suggests this regime may not exist for the WCM formulation with standard sigmoid parameters, or is extremely narrow and sensitive to perturbations.

## Available Tools

### Parameter Search Scripts

Three scripts are provided in `examples/`:

1. **`search_oscillatory_parameters.jl`**: Systematic search through connectivity and sigmoid parameters
   ```bash
   julia --project=. examples/search_oscillatory_parameters.jl
   ```

2. **`search_oscillatory_parameters_v2.jl`**: Explores baseline-subtracted sigmoids and time constant variations
   ```bash
   julia --project=. examples/search_oscillatory_parameters_v2.jl
   ```

3. **`search_oscillatory_parameters_theory.jl`**: Theory-guided search with differential time constants
   ```bash
   julia --project=. examples/search_oscillatory_parameters_theory.jl
   ```

### Helper Functions

The scripts provide reusable functions for parameter exploration:

- `create_ei_oscillatory_params()`: Create parameters with differential time constants
- `simulate_dynamics()`: Euler integration for testing parameter sets
- `analyze_oscillations()`: Detect and quantify oscillatory behavior
- `count_peaks()`: Count local maxima in time series

## Future Directions

1. **Bifurcation analysis**: Systematically map parameter space to find Hopf bifurcation points

2. **Spatial models**: Explore whether spatial coupling enables sustained oscillations

3. **Alternative nonlinearities**: Test piecewise-linear, threshold-linear, or other nonlinearities

4. **Noise-sustained oscillations**: Investigate whether stochastic forcing can sustain oscillations

5. **Network models**: Test whether heterogeneous networks of WCM units can produce sustained oscillations

6. **Experimental validation**: Compare with experimental data to determine if sustained oscillations are biologically plausible for cortical circuits

## References

1. Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory of the functional dynamics of cortical and thalamic nervous tissue. *Kybernetik*, 13(2), 55-80.

2. Ermentrout, G. B., & Terman, D. H. (2010). *Mathematical foundations of neuroscience* (Vol. 35). Springer.

3. Dayan, P., & Abbott, L. F. (2001). *Theoretical neuroscience: computational and mathematical modeling of neural systems*. MIT press.

## Conclusion

The parameter search revealed that sustained oscillations are difficult to achieve in the Wilson-Cowan point model with physiologically plausible parameters. The WCM 1973 "oscillatory mode" produces transient, damped oscillations. 

For truly sustained oscillations, consider:

- **Spatial models** with wave propagation
- **External forcing** or periodic stimulation
- **Alternative model types** designed for sustained oscillations
- **Model extensions** with adaptation or synaptic dynamics

The provided search scripts and analysis tools enable further exploration of the parameter space and can be adapted for different Wilson-Cowan formulations or related neural field models.
