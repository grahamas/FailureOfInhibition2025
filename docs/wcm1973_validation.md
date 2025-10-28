# Wilson-Cowan 1973 Validation and Parameter Mapping

This document provides a detailed mapping between the Wilson-Cowan 1973 paper and this implementation, enabling validation and replication of the paper's results.

## Reference

Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory of the functional dynamics of cortical and thalamic nervous tissue. *Kybernetik*, 13(2), 55-80.

## Overview

The 1973 paper describes three distinct dynamical modes that emerge from different parameter configurations:

1. **Active Transient Mode** - Characteristic of sensory neo-cortex
2. **Oscillatory Mode** - Characteristic of thalamus  
3. **Steady-State Mode** - Characteristic of archi- or prefrontal cortex

## Mathematical Formulation

### Paper's Equations (1.3.1 and 1.3.2)

The paper uses the following form for the Wilson-Cowan equations:

```
μ d⟨E⟩/dt = -⟨E⟩ + [1 - rₑ⟨E⟩] Sₑ[Qₑμ[bₑₑ⟨E⟩⊗fₑₑ - bᵢₑ⟨I⟩⊗fᵢₑ + ⟨P⟩]]

μ d⟨I⟩/dt = -⟨I⟩ + [1 - rᵢ⟨I⟩] Sᵢ[Qᵢμ[bₑᵢ⟨E⟩⊗fₑᵢ - bᵢᵢ⟨I⟩⊗fᵢᵢ + ⟨Q⟩]]
```

where:
- `⟨E⟩`, `⟨I⟩`: Time-averaged activities of excitatory and inhibitory populations
- `μ`: Membrane time constant (10 msec)
- `rₑ`, `rᵢ`: Refractory periods (1 msec)
- `Sⱼ`: Sigmoid nonlinearity function
- `Qⱼ`: Normalization constants (set to 1)
- `bⱼⱼ'`: Connectivity strength coefficients
- `fⱼⱼ'`: Spatial connectivity kernels (exponential decay)
- `⊗`: Spatial convolution operator
- `⟨P⟩`, `⟨Q⟩`: External stimuli to E and I populations

### This Implementation's Form

```
τᵢ dAᵢ/dt = -αᵢ Aᵢ + βᵢ (1 - Aᵢ) f(Sᵢ(t) + Cᵢ(A))
```

where:
- `Aᵢ`: Activity of population i
- `τᵢ`: Time constant
- `αᵢ`: Decay rate
- `βᵢ`: Saturation coefficient
- `f`: Nonlinearity (sigmoid)
- `Sᵢ(t)`: External stimulus (time-dependent)
- `Cᵢ(A)`: Recurrent connectivity input (activity-dependent)

## Parameter Mapping

### Direct Correspondences

| Paper Parameter | Implementation | Value/Notes |
|----------------|----------------|-------------|
| `μ` | `τ` | 10.0 msec (membrane time constant) |
| `rₑ`, `rᵢ` | implicit | 1.0 msec (limits max activity to 0.5) |
| Normalized decay | `α` | 1.0 (normalized in paper) |
| Normalized saturation | `β` | 1.0 (normalized in paper) |
| `Qₑ`, `Qᵢ` | implicit | 1.0 (normalization constants) |

### Sigmoid Parameters

The paper uses a logistic sigmoid with baseline subtraction:

```
Sⱼ(N) = [1 + exp(-vⱼ(N - θⱼ))]⁻¹ - [1 + exp(vⱼθⱼ)]⁻¹
```

Our implementation uses:

```
σ(x) = [1 + exp(-a(x - θ))]⁻¹
```

where `a = vⱼ` and `θ = θⱼ`.

**Note:** The baseline subtraction in the paper ensures S(0) = 0. Our implementation approximates this for qualitative behavior matching.

### Connectivity Parameters

The paper uses exponential connectivity kernels:

```
fⱼⱼ'(x) = bⱼⱼ' exp(-|x|/aⱼⱼ')
```

Our implementation uses Gaussian kernels as an approximation:

```
GaussianConnectivityParameter(amplitude, (length_scale,))
```

**Mapping:**
- Paper's `bⱼⱼ'` → `amplitude` parameter
- Paper's `aⱼⱼ'` → `length_scale` parameter (in spatial units)

**Sign Convention:**
- Excitatory connections: positive amplitude
- Inhibitory connections: negative amplitude

### Connectivity Matrix Indexing

Both the paper and implementation use the convention:
- `connectivity[i,j]` maps population j → population i

For 2 populations (E=1, I=2):
```
[E→E  I→E]
[E→I  I→I]
```

## Table 2 Parameters

The paper provides three parameter sets in Table 2. All spatial parameters are in micrometers (μm).

### Active Transient Mode

**Sigmoid Parameters:**
- `vₑ = 0.5`, `θₑ = 9.0` (excitatory sigmoid)
- `vᵢ = 0.3`, `θᵢ = 17.0` (inhibitory sigmoid)

**Connectivity Parameters:**
- `bₑₑ = 1.5`, `aₑₑ = 40.0 μm` (E → E, excitatory self-connection)
- `bᵢₑ = 1.35`, `aᵢₑ = 60.0 μm` (I → E, inhibitory to excitatory)
- `bₑᵢ = 1.35`, `aₑᵢ = 60.0 μm` (E → I, excitatory to inhibitory)
- `bᵢᵢ = 1.8`, `aᵢᵢ = 30.0 μm` (I → I, inhibitory self-connection)

**Behavior:**
- Brief localized stimuli elicit self-generated transient responses
- Activity continues to increase after stimulus cessation
- Eventually decays back to resting state
- Characteristic of sensory neo-cortex

### Oscillatory Mode

**Sigmoid Parameters:**
- `vₑ = 0.5`, `θₑ = 9.0` (excitatory sigmoid)
- `vᵢ = 1.0`, `θᵢ = 15.0` (inhibitory sigmoid - steeper than active transient)

**Connectivity Parameters:**
- `bₑₑ = 2.0`, `aₑₑ = 40.0 μm` (E → E)
- `bᵢₑ = 1.5`, `aᵢₑ = 60.0 μm` (I → E)
- `bₑᵢ = 1.5`, `aₑᵢ = 60.0 μm` (E → I)
- `bᵢᵢ = 0.1`, `aᵢᵢ = 20.0 μm` (I → I - much weaker than active transient)

**Key Differences from Active Transient:**
- Steeper inhibitory sigmoid (`vᵢ = 1.0` vs `0.3`)
- Weaker inhibitory-inhibitory coupling (`bᵢᵢ = 0.1` vs `1.8`)
- Stronger excitatory-inhibitory coupling (`bᵢₑ = 1.5` vs `1.35`)

**Behavior:**
- Sustained oscillations in response to adequate stimulation
- Oscillation frequency encodes stimulus intensity
- Edge enhancement occurs for sufficiently wide stimuli
- Characteristic of thalamus

### Steady-State Mode

**Sigmoid Parameters:**
- `vₑ = 0.5`, `θₑ = 9.0` (excitatory sigmoid)
- `vᵢ = 0.3`, `θᵢ = 17.0` (inhibitory sigmoid - same as active transient)

**Connectivity Parameters:**
- `bₑₑ = 2.0`, `aₑₑ = 40.0 μm` (E → E - stronger than active transient)
- `bᵢₑ = 1.35`, `aᵢₑ = 60.0 μm` (I → E)
- `bₑᵢ = 1.35`, `aₑᵢ = 60.0 μm` (E → I)
- `bᵢᵢ = 1.8`, `aᵢᵢ = 30.0 μm` (I → I)

**Key Difference from Active Transient:**
- Only differs in `bₑₑ` (2.0 vs 1.5)
- Stronger excitatory-excitatory coupling

**Behavior:**
- Spatially inhomogeneous stable steady states
- Can retain contour information about prior stimuli
- Activity patterns persist after stimulus removal
- Characteristic of archi- or prefrontal cortex

## Fixed Parameters

The following parameters are held constant across all three modes:

- **Membrane time constant** (`μ`): 10 msec
  - Gives reasonable oscillation frequencies (~25-100 Hz)
  - Matches typical cortical neuron time constants

- **Refractory periods** (`rₑ`, `rᵢ`): 1 msec
  - Absolute refractory period duration
  - Limits maximum activity to 0.5 (50% of population)

- **Excitatory threshold** (`θₑ`): ~9-10 units
  - Based on observations of motoneuron and pyramidal cell thresholds
  - Kept constant to isolate effects of connectivity changes

- **Length constants** (`aⱼⱼ'`): 20-60 μm
  - Based on Sholl's data for cat striate cortex
  - Within biologically plausible range for lateral connections

## Implementation Usage

### Point Model (Non-Spatial)

Use `create_point_model_wcm1973(mode)` from `test/test_wcm1973_validation.jl`:

```julia
using FailureOfInhibition2025
include("test/test_wcm1973_validation.jl")

# Create parameters for desired mode
params = create_point_model_wcm1973(:active_transient)  # or :oscillatory, :steady_state

# Initial condition (1, 2) array for 2 populations
A₀ = reshape([0.1, 0.1], 1, 2)

# Compute derivatives
dA = zeros(1, 2)
wcm1973!(dA, A₀, params, 0.0)
```

### Spatial Model (1D)

Use `create_wcm1973_parameters(mode)` from `test/test_wcm1973_validation.jl`:

```julia
# Create spatial parameters
params = create_wcm1973_parameters(:active_transient)

# Initial condition (N_points, 2) array
n_points = size(params.lattice)[1]  # 101 points
A₀ = 0.1 .* ones(n_points, 2)

# Compute derivatives
dA = zeros(n_points, 2)
wcm1973!(dA, A₀, params, 0.0)
```

## Validation Tests

The file `test/test_wcm1973_validation.jl` provides comprehensive tests that validate:

1. **Parameter Construction** - All three modes construct correctly
2. **Point Model Dynamics** - Non-spatial models work as expected
3. **Spatial Model Dynamics** - Spatial connectivity propagates correctly
4. **Parameter Relationships** - Differences between modes are correct
5. **Basic Behavior** - Resting state stability, decay dynamics, etc.

Run the validation tests:

```bash
julia --project=. test/test_wcm1973_validation.jl
```

Or as part of the full test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Future Work: Quantitative Validation

To fully replicate the paper's results, the following tests should be added:

### Figure 3: Temporal Summation and Latency
- Test that activity peaks after stimulus cessation
- Verify latency decreases with increasing stimulus intensity
- Validate temporal summation effects

### Figure 4: Spatial Summation
- Show activity depends on both stimulus width and intensity
- Larger stimuli should produce different peak responses

### Figure 5: Edge Enhancement
- Verify that edges are enhanced in spatial activity profiles
- Check that enhancement has characteristic latency (~15 msec)

### Figure 6: Size-Intensity Effect
- Show that edge separation increases with stimulus intensity
- If this disparity is used to measure stimulus size, predict that more intense stimuli appear larger

### Figure 7: Oscillatory Mode Dynamics
- Verify sustained oscillations above threshold
- Show edge enhancement in oscillatory mode
- Narrow vs wide stimulus differences

### Table 4: Limit Cycle Frequency
- Measure oscillation frequency as function of stimulus intensity
- Verify frequency increases with intensity in predicted manner

### Steady-State Mode Patterns
- Show formation of stable spatial patterns
- Verify patterns persist after stimulus removal
- Demonstrate retention of contour information

## Notes on Approximations

1. **Gaussian vs Exponential Kernels**: The paper uses exponential connectivity kernels, we use Gaussian approximations. This is common in neural field modeling and gives qualitatively similar results.

2. **Sigmoid Baseline**: The paper subtracts the baseline from the sigmoid so S(0) = 0 exactly. We use the standard sigmoid which has small but non-zero values at zero input. This gives similar qualitative behavior.

3. **Spatial Discretization**: The paper's equations are continuous in space. We discretize using a lattice with finite spacing (~10 μm). This is necessary for numerical simulation.

4. **Time Units**: The paper uses milliseconds throughout. Our implementation is agnostic to units but documentation assumes milliseconds to match the paper.

## References

For implementation details, see:
- `src/models.jl` - Wilson-Cowan model implementation
- `src/connect.jl` - Connectivity implementation
- `src/nonlinearity.jl` - Sigmoid nonlinearity implementation
- `test/test_wcm1973_validation.jl` - Validation tests
- `examples/example_wcm1973_modes.jl` - Usage examples

For the mathematical theory, see the original paper:
- Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory of the functional dynamics of cortical and thalamic nervous tissue. *Kybernetik*, 13(2), 55-80.
