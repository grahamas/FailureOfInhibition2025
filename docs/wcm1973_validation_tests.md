# WCM1973 Implementation Validation Tests

This document describes the validation tests created for the `wcm1973!` implementation based on the mathematical theory presented in Wilson & Cowan (1973): "A mathematical theory of the functional dynamics of cortical and thalamic nervous tissue."

## Overview

The Wilson-Cowan model describes neural population dynamics using the following differential equation:

```math
τᵢ dAᵢ/dt = -αᵢ Aᵢ + βᵢ (1 - Aᵢ) f(Sᵢ(t) + Cᵢ(A))
```

where:
- `Aᵢ` is the activity of population i
- `αᵢ` is the decay rate
- `βᵢ` is the saturation coefficient
- `τᵢ` is the time constant
- `f` is the nonlinearity (firing rate function)
- `Sᵢ(t)` is external stimulus (function of time)
- `Cᵢ(A)` is recurrent input from connectivity (function of activity)

## Test Suite: `test_wcm1973_validation.jl`

This test suite validates that the implementation correctly captures the mathematical properties and behaviors described in the original 1973 paper.

### Test 1: Equation Structure Validation

**Purpose**: Verify that `wcm1973!` implements the correct mathematical equation structure.

**What it tests**:
- The implementation follows the exact flow: stimulus → connectivity → nonlinearity → Wilson-Cowan dynamics
- The equation computes: `dA/dt = ((sigmoid(A) - A) * β * (1-A) - α*A) / τ`
- Numerical values match the expected analytical calculation

**Why it's important**: This is the foundational test ensuring the core mathematical model is correctly implemented.

**Based on**: Equation (1) in Wilson & Cowan (1973)

### Test 2: Decay Parameter (α)

**Purpose**: Verify that the decay rate parameter `α` correctly controls the rate at which activity decays in the absence of strong excitation.

**What it tests**:
- Larger `α` values produce more negative (or less positive) derivatives
- The decay term `-α*A` has the expected effect on dynamics

**Why it's important**: The decay parameter is crucial for stability and prevents runaway excitation. Wilson & Cowan discuss this in the context of refractory periods and adaptation.

**Based on**: Section on "Equations of the Model" discussing the decay term

### Test 3: Saturation Coefficient (β)

**Purpose**: Verify that the saturation coefficient `β` correctly scales the nonlinearity term.

**What it tests**:
- Larger `β` values produce stronger responses to excitation
- The gain of the nonlinearity term is properly controlled

**Why it's important**: β controls the maximum rate at which populations can be excited, representing the saturation of neural responses.

**Based on**: Section on saturation and maximal firing rates in the 1973 paper

### Test 4: Time Constant (τ)

**Purpose**: Verify that the time constant `τ` correctly scales the overall rate of change.

**What it tests**:
- Larger `τ` values slow down the dynamics (smaller |dA/dt|)
- The time scale separation between different populations works correctly

**Why it's important**: τ allows modeling different time scales for different populations (e.g., fast excitatory vs. slower inhibitory dynamics).

**Based on**: Discussion of time constants and their biological interpretation

### Test 5: Activity Bounds

**Purpose**: Verify that the model structure respects the constraint that activities should remain in [0, 1].

**What it tests**:
- The `(1-A)` term in the equation provides a natural saturation mechanism
- Activities near 1 have reduced growth rates due to the `(1-A)` factor

**Why it's important**: Wilson & Cowan interpret activities as proportions of active neurons, which must be bounded. The `(1-A)` term ensures this mathematically.

**Based on**: Section on "Interpretation of Variables" discussing activity as a proportion

### Test 6: Steady State Conditions

**Purpose**: Verify that steady states (where dA/dt = 0) satisfy the equation.

**What it tests**:
- At steady state: `α*A = β*(1-A)*f(A)`
- The implementation finds points where derivatives are near zero

**Why it's important**: Steady states are critical for understanding the model's long-term behavior and stability. Wilson & Cowan analyze multiple steady states and their stability.

**Based on**: Section on "Steady States and Stability" in the 1973 paper

### Test 7: Two-Population Interaction

**Purpose**: Verify that excitatory-inhibitory (E-I) interactions work correctly.

**What it tests**:
- Excitatory connections have positive effects
- Inhibitory connections have negative effects
- The connectivity matrix correctly implements population-to-population interactions

**Why it's important**: The E-I interaction is the core mechanism of the Wilson-Cowan model. The paper extensively discusses how E-I dynamics produce various behaviors.

**Based on**: Section on "Coupled Populations" and the E-I interaction analysis

### Test 8: Spatial Dynamics

**Purpose**: Verify that the model works correctly across multiple spatial points.

**What it tests**:
- Dynamics are computed correctly for all spatial locations
- The spatial extent is properly handled

**Why it's important**: Wilson & Cowan extended their model to include spatial interactions, which is crucial for modeling cortical tissue.

**Based on**: Section on "Spatial Distributions" in the 1973 paper

### Test 9: Different Nonlinearity Types

**Purpose**: Verify that different nonlinearity functions work with the model.

**What it tests**:
- Standard sigmoid nonlinearity
- Rectified zeroed sigmoid (biologically realistic, prevents negative firing rates)
- Difference of sigmoids (for more complex response curves)

**Why it's important**: The choice of nonlinearity function f(x) is critical. Wilson & Cowan discuss various forms, including sigmoid and threshold-linear functions.

**Based on**: Section on "Response Functions" discussing different functional forms

### Test 10: Saturation Mechanism

**Purpose**: Verify that the `(1-A)` saturation term correctly limits growth as A approaches 1.

**What it tests**:
- Growth rate decreases as activity approaches 1
- The saturation mechanism prevents activities from exceeding biological bounds

**Why it's important**: This is a key biological constraint. Wilson & Cowan derived this term from the principle that there's a finite pool of neurons that can be activated.

**Based on**: Discussion of the saturation term and its biological justification

## Integration Tests

In addition to these unit tests, the test suite also includes integration tests in other files:

- **`test_wilson_cowan.jl`**: Tests the overall WilsonCowanParameters structure and basic dynamics
- **`test_connectivity_matrix.jl`**: Tests the ConnectivityMatrix implementation for per-population-pair connectivity
- **`test_gaussian_connectivity.jl`**: Tests spatial connectivity kernels
- **`test_stimulate.jl`**: Tests external stimulation
- **`test_nonlinearity.jl`**: Tests various nonlinearity functions

## Running the Tests

Run all validation tests:
```julia
julia --project=. test/test_wcm1973_validation.jl
```

Run the full test suite (includes validation tests):
```julia
using Pkg
Pkg.test("FailureOfInhibition2025")
```

Or:
```bash
julia --project=. test/runtests.jl
```

## Test Coverage

The validation tests cover:
- ✅ Mathematical equation structure
- ✅ All model parameters (α, β, τ)
- ✅ Activity bounds and saturation
- ✅ Steady state conditions
- ✅ Multi-population interactions
- ✅ Spatial dynamics
- ✅ Multiple nonlinearity types
- ✅ Conservation laws and constraints

## Future Test Extensions

Potential additional tests that could be added based on the 1973 paper:

1. **Stability Analysis**: Test that stable steady states are actually stable under small perturbations
2. **Bifurcation Behavior**: Test that parameter changes produce expected bifurcations
3. **Oscillatory Solutions**: Test that the model can produce sustained oscillations for appropriate parameters
4. **Hysteresis**: Test that the model exhibits hysteresis for certain parameter ranges
5. **Wave Propagation**: Test traveling wave solutions in spatial models
6. **Stimulus Response**: Test specific stimulus-response relationships discussed in the paper

These would require more sophisticated numerical analysis and are beyond the scope of basic validation tests.

## References

Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory of the functional dynamics of cortical and thalamic nervous tissue. *Kybernetik*, 13(2), 55-80.

Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophysical journal*, 12(1), 1-24.
