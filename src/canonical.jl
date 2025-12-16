"""
Canonical Wilson-Cowan model parameter sets from Wilson & Cowan (1973).

This module provides functions to create Wilson-Cowan model parameters matching the
three dynamical modes described in:
Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory of the functional 
dynamics of cortical and thalamic nervous tissue. Kybernetik, 13(2), 55-80.

The three modes are:
1. Active Transient Mode - characteristic of sensory neo-cortex
2. Oscillatory Mode - characteristic of thalamus
3. Steady-State Mode - characteristic of archi- or prefrontal cortex
"""

"""
    create_wcm1973_parameters(mode::Symbol; lattice=nothing)

Create WilsonCowanParameters matching Table 2 from Wilson & Cowan 1973.

# Arguments
- `mode`: One of :active_transient, :oscillatory, or :steady_state
- `lattice`: Spatial lattice (defaults to 1D lattice if not provided)

# Returns
WilsonCowanParameters{2} configured for the specified mode

# Parameter Mapping from Paper to Implementation

The paper uses the following equation form (Equations 1.3.1 and 1.3.2):
```
μ d<E>/dt = -<E> + [1 - rₑ<E>] Sₑ[Qₑμ[bₑₑ<E>⊗fₑₑ - bᵢₑ<I>⊗fᵢₑ + <P>]]
μ d<I>/dt = -<I> + [1 - rᵢ<I>] Sᵢ[Qᵢμ[bₑᵢ<E>⊗fₑᵢ - bᵢᵢ<I>⊗fᵢᵢ + <Q>]]
```

Our implementation uses:
```
τᵢ dAᵢ/dt = -αᵢ Aᵢ + βᵢ (1 - Aᵢ) f(Sᵢ(t) + Cᵢ(A))
```

Parameter mappings:
- τ (our implementation) = μ (paper) = 10 msec (membrane time constant)
- α (our implementation) = 1 (paper normalizes decay to unity)
- β (our implementation) = 1 (paper assumes saturation coefficient = 1)
- rₑ, rᵢ (paper) = 1 msec (refractory period, limits max activity to 0.5)
- Qₑ, Qᵢ (paper) = 1 (normalization constants)
- vₑ, vᵢ (paper) = slope of sigmoid at threshold
- θₑ, θᵢ (paper) = threshold values for sigmoid
- bⱼⱼ' (paper) = connectivity strengths (weights)
- aⱼⱼ' (paper) = connectivity length constants (μm)

# Connectivity Functions

Paper uses exponential connectivity:
```
fⱼⱼ'(x) = bⱼⱼ' exp(-|x|/aⱼⱼ')
```

Note: The paper's connectivity includes both the weight (b) and spatial kernel (exp).
In our implementation, we separate these:
- GaussianConnectivityParameter takes amplitude and length scale
- For exponential decay, we approximate with Gaussian (common in neural field models)

# Table 2 Parameters (from Wilson & Cowan 1973)

All length constants (aⱼⱼ') are in micrometers (μm).

## Active Transient Mode
- vₑ = 0.5, θₑ = 9.0
- vᵢ = 0.3, θᵢ = 17.0  
- bₑₑ = 1.5, aₑₑ = 40.0 μm
- bᵢₑ = 1.35, aᵢₑ = 60.0 μm
- bₑᵢ = 1.35, aₑᵢ = 60.0 μm
- bᵢᵢ = 1.8, aᵢᵢ = 30.0 μm

## Oscillatory Mode
- vₑ = 0.5, θₑ = 9.0
- vᵢ = 1.0, θᵢ = 15.0
- bₑₑ = 2.0, aₑₑ = 40.0 μm
- bᵢₑ = 1.5, aᵢₑ = 60.0 μm
- bₑᵢ = 1.5, aₑᵢ = 60.0 μm
- bᵢᵢ = 0.1, aᵢᵢ = 20.0 μm

## Steady-State Mode
- vₑ = 0.5, θₑ = 9.0
- vᵢ = 0.3, θᵢ = 17.0
- bₑₑ = 2.0, aₑₑ = 40.0 μm (note: increased from active transient)
- bᵢₑ = 1.35, aᵢₑ = 60.0 μm
- bₑᵢ = 1.35, aₑᵢ = 60.0 μm
- bᵢᵢ = 1.8, aᵢᵢ = 30.0 μm

# Fixed Parameters (used in all modes)
- μ = 10 msec (membrane time constant)
- rₑ = rᵢ = 1 msec (refractory periods)
- Max activity = [1 + r]⁻¹ ≈ 0.5 (due to refractory period)
"""
function create_wcm1973_parameters(mode::Symbol; lattice=nothing)
    # Create default lattice if not provided
    # Use 1D lattice with extent and spacing matching the paper
    # Paper uses spatial coordinates in micrometers
    if lattice === nothing
        # Create a 1D lattice with 1mm extent (1000 μm) and fine spatial resolution
        # 101 points gives 10 μm spacing
        lattice = CompactLattice(extent=(1000.0,), n_points=(101,))
    end
    
    # Get parameters based on mode
    if mode == :active_transient
        # Table 2, Active Transient Mode
        vₑ, θₑ = 0.5, 9.0
        vᵢ, θᵢ = 0.3, 17.0
        bₑₑ, aₑₑ = 1.5, 40.0
        bᵢₑ, aᵢₑ = 1.35, 60.0
        bₑᵢ, aₑᵢ = 1.35, 60.0
        bᵢᵢ, aᵢᵢ = 1.8, 30.0
    elseif mode == :oscillatory
        # Table 2, Oscillatory Mode
        vₑ, θₑ = 0.5, 9.0
        vᵢ, θᵢ = 1.0, 15.0
        bₑₑ, aₑₑ = 2.0, 40.0
        bᵢₑ, aᵢₑ = 1.5, 60.0
        bₑᵢ, aₑᵢ = 1.5, 60.0
        bᵢᵢ, aᵢᵢ = 0.1, 20.0
    elseif mode == :steady_state
        # Table 2, Steady-State Mode
        vₑ, θₑ = 0.5, 9.0
        vᵢ, θᵢ = 0.3, 17.0
        bₑₑ, aₑₑ = 2.0, 40.0  # Note: increased bₑₑ compared to active transient
        bᵢₑ, aᵢₑ = 1.35, 60.0
        bₑᵢ, aₑᵢ = 1.35, 60.0
        bᵢᵢ, aᵢᵢ = 1.8, 30.0
    else
        error("Unknown mode: $mode. Use :active_transient, :oscillatory, or :steady_state")
    end
    
    # Create nonlinearity for each population
    # Paper uses logistic sigmoid: S(N) = [1 + exp(-v(N - θ))]⁻¹ - [1 + exp(vθ)]⁻¹
    # Our SigmoidNonlinearity uses: σ(x) = [1 + exp(-a(x - θ))]⁻¹
    # We need to handle the baseline subtraction separately or accept the approximation
    # For now, use the standard sigmoid which is close enough for qualitative behavior
    nonlinearity_e = SigmoidNonlinearity(a=vₑ, θ=θₑ)
    nonlinearity_i = SigmoidNonlinearity(a=vᵢ, θ=θᵢ)
    nonlinearity = (nonlinearity_e, nonlinearity_i)
    
    # Create connectivity matrix
    # Paper uses exponential kernels, we approximate with Gaussian
    # For Gaussian with std σ, approximate exponential with length a: σ ≈ a/√2
    # However, for better matching, we can use the length constant directly
    
    # Connectivity indexing: connectivity[i,j] maps j → i
    # E=1, I=2
    # [E→E  I→E]
    # [E→I  I→I]
    
    # Note: In the paper, excitatory connections are positive, inhibitory are negative
    # The b parameters include this sign
    conn_ee = GaussianConnectivityParameter(bₑₑ, (aₑₑ,))    # E → E (excitatory)
    conn_ei = GaussianConnectivityParameter(-bᵢₑ, (aᵢₑ,))   # I → E (inhibitory)
    conn_ie = GaussianConnectivityParameter(bₑᵢ, (aₑᵢ,))    # E → I (excitatory)
    conn_ii = GaussianConnectivityParameter(-bᵢᵢ, (aᵢᵢ,))   # I → I (inhibitory)
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    # Fixed parameters from the paper
    μ = 10.0  # msec, membrane time constant
    rₑ = rᵢ = 1.0  # msec, refractory period
    
    # Create parameters
    # In our formulation:
    # τ corresponds to μ in the paper
    # α = 1 (normalized decay)
    # β = 1 (normalized saturation)
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.0),      # Normalized decay rates
        β = (1.0, 1.0),      # Normalized saturation coefficients  
        τ = (μ, μ),          # Membrane time constants
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = nothing,   # Set stimulus separately for each test
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    return params
end

"""
    load_optimized_wcm1973_parameters(variant::Symbol=:oscillatory_optimized)

Load optimized WCM1973 parameters from JSON file for a specific variant.

# Arguments
- `variant`: Optimization variant to load (default: :oscillatory_optimized)
  Future variants might include :active_transient_optimized, :steady_state_optimized, etc.

# Returns
Dictionary with parameter values and metadata

# Notes
The file naming convention is: `wcm1973_{variant}.json`
For :oscillatory_optimized, the file is `wcm1973_oscillatory_optimized.json`
"""
function load_optimized_wcm1973_parameters(variant::Symbol=:oscillatory_optimized)
    # Construct variant-specific filename
    variant_filename = "wcm1973_$(variant).json"
    json_path = joinpath(@__DIR__, "..", "data", variant_filename)
    
    if !isfile(json_path)
        error("Optimized parameters file not found at: $json_path\n" *
              "Run the optimization script first: julia --project=. scripts/optimize_oscillation_parameters.jl\n" *
              "Or ensure the file is named correctly: $variant_filename")
    end
    
    # Try to load with JSON if available
    JSON_mod = nothing
    try
        # Use Base.invokelatest to avoid world age issues
        JSON_mod = Base.require(Main, :JSON)
    catch e
        if isa(e, ArgumentError)
            error("JSON package is required to load optimized parameters. Run: using Pkg; Pkg.add(\"JSON\")")
        else
            error("Failed to load JSON package: $(typeof(e))")
        end
    end
    
    # Now parse the file with dicttype to get plain Dict
    try
        data = Base.invokelatest(JSON_mod.parsefile, json_path; dicttype=Dict)
        return data
    catch e
        error("Failed to load optimized parameters from $json_path: $(typeof(e)) - Check that the file is valid JSON")
    end
end

"""
    create_point_model_wcm1973(mode::Symbol)

Create a point (non-spatial) model matching Wilson & Cowan 1973.

This creates a simplified version using PointLattice and ScalarConnectivity,
which corresponds to the spatially localized aggregate equations (2.0.1 and 2.0.2)
in the paper.

# Arguments
- `mode`: One of :active_transient, :oscillatory, :steady_state, or optimized variants
  Canonical modes: :active_transient, :oscillatory, :steady_state (from WCM 1973 Table 2)
  Optimized variants: Prefix with "optimized_", e.g., :optimized_oscillatory

# Returns
WilsonCowanParameters{2} configured for the specified mode with PointLattice

# Examples
```julia
# Canonical modes from paper
params_osc = create_point_model_wcm1973(:oscillatory)

# Optimized variant (loads from JSON)
params_opt = create_point_model_wcm1973(:optimized_oscillatory)
```
"""
function create_point_model_wcm1973(mode::Symbol)
    # Create point lattice (zero-dimensional space)
    lattice = PointLattice()
    
    # Check if this is an optimized variant (starts with "optimized_")
    mode_str = string(mode)
    is_optimized = startswith(mode_str, "optimized_")
    
    if is_optimized
        # Extract base mode (e.g., "oscillatory" from "optimized_oscillatory")
        base_mode_str = replace(mode_str, "optimized_" => "")
        variant_symbol = Symbol(base_mode_str * "_optimized")
        
        # Load optimized parameters from file
        opt_data = load_optimized_wcm1973_parameters(variant_symbol)
        
        # Load parameters from JSON file
        params_dict = opt_data["parameters"]
        vₑ = params_dict["nonlinearity"]["v_e"]
        θₑ = params_dict["nonlinearity"]["theta_e"]
        vᵢ = params_dict["nonlinearity"]["v_i"]
        θᵢ = params_dict["nonlinearity"]["theta_i"]
        bₑₑ = params_dict["connectivity"]["b_ee"]
        bᵢₑ = params_dict["connectivity"]["b_ei"]  # Already includes sign
        bₑᵢ = params_dict["connectivity"]["b_ie"]
        bᵢᵢ = params_dict["connectivity"]["b_ii"]  # Already includes sign
        τₑ = params_dict["tau_e"]
        τᵢ = params_dict["tau_i"]
    elseif mode == :active_transient
        vₑ, θₑ = 0.5, 9.0
        vᵢ, θᵢ = 0.3, 17.0
        bₑₑ = 1.5
        bᵢₑ = 1.35
        bₑᵢ = 1.35
        bᵢᵢ = 1.8
        τₑ, τᵢ = 10.0, 10.0
    elseif mode == :oscillatory
        vₑ, θₑ = 0.5, 9.0
        vᵢ, θᵢ = 1.0, 15.0
        bₑₑ = 2.0
        bᵢₑ = 1.5
        bₑᵢ = 1.5
        bᵢᵢ = 0.1
        τₑ, τᵢ = 10.0, 10.0
    elseif mode == :steady_state
        vₑ, θₑ = 0.5, 9.0
        vᵢ, θᵢ = 0.3, 17.0
        bₑₑ = 2.0
        bᵢₑ = 1.35
        bₑᵢ = 1.35
        bᵢᵢ = 1.8
        τₑ, τᵢ = 10.0, 10.0
    else
        error("Unknown mode: $mode. Use :active_transient, :oscillatory, :steady_state, or :optimized_{base_mode}")
    end
    
    # Create nonlinearity - separate for E and I populations
    nonlinearity_e = SigmoidNonlinearity(a=vₑ, θ=θₑ)
    nonlinearity_i = SigmoidNonlinearity(a=vᵢ, θ=θᵢ)
    nonlinearity = (nonlinearity_e, nonlinearity_i)
    
    # Create scalar connectivity for point model
    conn_ee = ScalarConnectivity(bₑₑ)
    conn_ei = ScalarConnectivity(-bᵢₑ)
    conn_ie = ScalarConnectivity(bₑᵢ)
    conn_ii = ScalarConnectivity(-bᵢᵢ)
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    # Create parameters
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.0),
        β = (1.0, 1.0),
        τ = (τₑ, τᵢ),
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    return params
end

"""
Canonical model parameter sets from prototypes.

These models are adapted from TravelingWaveSimulations prototypes.jl:
https://github.com/grahamas/TravelingWaveSimulations/blob/c15473966d764a5f4be60f8215575fac1f7f1bee/src/prototypes/prototypes.jl

The prototypes represent various dynamical regimes including:
- Harris & Ermentrout 2018 traveling wave model
- Full dynamics with monotonic and blocking inhibition
- Oscillating pulse dynamics
- Propagating patterns on torus geometries
"""

"""
    create_harris_ermentrout_parameters(; lattice=nothing, kwargs...)

Create WilsonCowanParameters matching Harris & Ermentrout 2018 traveling wave model.

This parameterization is designed to produce propagating traveling waves in 
excitatory-inhibitory neural field models.

# Reference
Harris, K. D., & Ermentrout, G. B. (2018). Bifurcations in the Wilson-Cowan equations 
with distributed delays, and application to the stability of cortical spreading depression.
SIAM Journal on Applied Dynamical Systems, 17(1), 501-520.

# Keyword Arguments
- `lattice`: Spatial lattice (defaults to 1D periodic lattice if not provided)
- `Aee`, `See`: Excitatory-to-excitatory amplitude and spread (default: 1.0, 25.0)
- `Aii`, `Sii`: Inhibitory-to-inhibitory amplitude and spread (default: 0.25, 27.0)
- `Aie`, `Sie`: Excitatory-to-inhibitory amplitude and spread (default: 1.0, 25.0)
- `Aei`, `Sei`: Inhibitory-to-excitatory amplitude and spread (default: 1.5, 27.0)
- `aE`, `θE`: Excitatory nonlinearity slope and threshold (default: 50.0, 0.125)
- `aI`, `θI`: Inhibitory nonlinearity slope and threshold (default: 50.0, 0.4)
- `α`: Decay rates for (E, I) (default: (1.0, 1.0))
- `β`: Saturation coefficients for (E, I) (default: (1.0, 1.0))
- `τ`: Time constants for (E, I) (default: (1.0, 0.4))

# Returns
WilsonCowanParameters{2} configured for Harris-Ermentrout traveling waves

# Example
```julia
params = create_harris_ermentrout_parameters()
A₀ = 0.1 .+ 0.01 .* rand(size(params.lattice.n_points)..., 2)
sol = solve_model(A₀, (0.0, 300.0), params)
```
"""
function create_harris_ermentrout_parameters(;
    lattice=nothing,
    Aee=1.0, See=25.0,
    Aii=0.25, Sii=27.0,
    Aie=1.0, Sie=25.0,
    Aei=1.5, Sei=27.0,
    aE=50.0, θE=0.125,
    aI=50.0, θI=0.4,
    α=(1.0, 1.0),
    β=(1.0, 1.0),
    τ=(1.0, 0.4))
    
    # Create default lattice if not provided
    # Use 1D periodic lattice with parameters from prototype
    if lattice === nothing
        lattice = PeriodicLattice(extent=(1400.0,), n_points=(512,))
    end
    
    # Create nonlinearity for each population
    nonlinearity_e = SigmoidNonlinearity(a=aE, θ=θE)
    nonlinearity_i = SigmoidNonlinearity(a=aI, θ=θI)
    nonlinearity = (nonlinearity_e, nonlinearity_i)
    
    # Create connectivity matrix
    conn_ee = GaussianConnectivityParameter(Aee, (See,))
    conn_ei = GaussianConnectivityParameter(-Aei, (Sei,))
    conn_ie = GaussianConnectivityParameter(Aie, (Sie,))
    conn_ii = GaussianConnectivityParameter(-Aii, (Sii,))
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    # Create parameters
    params = WilsonCowanParameters{2}(
        α = α,
        β = β,
        τ = τ,
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    return params
end

"""
    create_harris_ermentrout_rectified_parameters(; kwargs...)

Create Harris-Ermentrout parameters with rectified sigmoid nonlinearities.

This is a variant that uses RectifiedZeroedSigmoidNonlinearity instead of
simple sigmoids for more biologically realistic dynamics (firing rates cannot be negative).

See `create_harris_ermentrout_parameters` for parameter descriptions.
"""
function create_harris_ermentrout_rectified_parameters(;
    aE=50.0, θE=0.125,
    aI=50.0, θI=0.4,
    kwargs...)
    
    # Create rectified nonlinearities
    nonlinearity_e = RectifiedZeroedSigmoidNonlinearity(a=aE, θ=θE)
    nonlinearity_i = RectifiedZeroedSigmoidNonlinearity(a=aI, θ=θI)
    
    # Create base parameters
    base_params = create_harris_ermentrout_parameters(; aE=aE, θE=θE, aI=aI, θI=θI, kwargs...)
    
    # Replace nonlinearity with rectified version
    return WilsonCowanParameters{2}(
        α = base_params.α,
        β = base_params.β,
        τ = base_params.τ,
        connectivity = base_params.connectivity,
        nonlinearity = (nonlinearity_e, nonlinearity_i),
        stimulus = base_params.stimulus,
        lattice = base_params.lattice,
        pop_names = base_params.pop_names
    )
end

"""
    create_full_dynamics_monotonic_parameters(; lattice=nothing, kwargs...)

Create parameters for full dynamics model with monotonic inhibition.

This parameterization uses more realistic time constants and decay rates
compared to the Harris-Ermentrout model, suitable for studying full neural
field dynamics including spread and propagation phenomena.

# Keyword Arguments
- `lattice`: Spatial lattice (defaults to 1D periodic lattice if not provided)
- `Aee`, `See`: Excitatory-to-excitatory amplitude and spread (default: 1.0, 25.0)
- `Aii`, `Sii`: Inhibitory-to-inhibitory amplitude and spread (default: 0.25, 27.0)
- `Aie`, `Sie`: Excitatory-to-inhibitory amplitude and spread (default: 1.0, 25.0)
- `Aei`, `Sei`: Inhibitory-to-excitatory amplitude and spread (default: 1.5, 27.0)
- `aE`, `θE`: Excitatory nonlinearity slope and threshold (default: 50.0, 0.125)
- `aI`, `θI`: Inhibitory nonlinearity slope and threshold (default: 50.0, 0.2)
- `αE`, `αI`: Decay rates for E and I (default: 0.4, 0.7)
- `β`: Saturation coefficients for (E, I) (default: (1.0, 1.0))
- `τE`, `τI`: Time constants for E and I (default: 1.0, 0.4)

# Returns
WilsonCowanParameters{2} configured for full dynamics with monotonic inhibition
"""
function create_full_dynamics_monotonic_parameters(;
    lattice=nothing,
    Aee=1.0, See=25.0,
    Aii=0.25, Sii=27.0,
    Aie=1.0, Sie=25.0,
    Aei=1.5, Sei=27.0,
    aE=50.0, θE=0.125,
    aI=50.0, θI=0.2,
    αE=0.4, αI=0.7,
    β=(1.0, 1.0),
    τE=1.0, τI=0.4)
    
    # Create default lattice if not provided
    if lattice === nothing
        lattice = PeriodicLattice(extent=(1400.0,), n_points=(512,))
    end
    
    # Create nonlinearity for each population
    nonlinearity_e = SigmoidNonlinearity(a=aE, θ=θE)
    nonlinearity_i = SigmoidNonlinearity(a=aI, θ=θI)
    nonlinearity = (nonlinearity_e, nonlinearity_i)
    
    # Create connectivity matrix
    conn_ee = GaussianConnectivityParameter(Aee, (See,))
    conn_ei = GaussianConnectivityParameter(-Aei, (Sei,))
    conn_ie = GaussianConnectivityParameter(Aie, (Sie,))
    conn_ii = GaussianConnectivityParameter(-Aii, (Sii,))
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    # Create parameters
    params = WilsonCowanParameters{2}(
        α = (αE, αI),
        β = β,
        τ = (τE, τI),
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    return params
end

"""
    create_full_dynamics_blocking_parameters(; kwargs...)

Create parameters for full dynamics model with blocking (non-monotonic) inhibition.

This parameterization uses a difference of sigmoids for the inhibitory population,
creating a non-monotonic response characteristic of failure of inhibition dynamics.

# Keyword Arguments
- `firing_aI`, `firing_θI`: Activating sigmoid parameters for inhibition (default: 50.0, 0.2)
- `blocking_aI`, `blocking_θI`: Blocking sigmoid parameters for inhibition (default: 50.0, 0.5)
- Other parameters same as `create_full_dynamics_monotonic_parameters`

# Returns
WilsonCowanParameters{2} configured for full dynamics with blocking inhibition
"""
function create_full_dynamics_blocking_parameters(;
    firing_aI=50.0, firing_θI=0.2,
    blocking_aI=50.0, blocking_θI=0.5,
    aE=50.0, θE=0.125,
    αE=0.4, αI=0.7,
    kwargs...)
    
    # Get base parameters with monotonic inhibition
    base_params = create_full_dynamics_monotonic_parameters(;
        aE=aE, θE=θE, aI=firing_aI, θI=firing_θI,
        αE=αE, αI=αI, kwargs...)
    
    # Create blocking inhibition nonlinearity
    nonlinearity_e = RectifiedZeroedSigmoidNonlinearity(a=aE, θ=θE)
    nonlinearity_i = DifferenceOfSigmoidsNonlinearity(
        a_activating=firing_aI,
        θ_activating=firing_θI,
        a_failing=blocking_aI,
        θ_failing=blocking_θI
    )
    
    # Replace nonlinearity with blocking version
    return WilsonCowanParameters{2}(
        α = base_params.α,
        β = base_params.β,
        τ = base_params.τ,
        connectivity = base_params.connectivity,
        nonlinearity = (nonlinearity_e, nonlinearity_i),
        stimulus = base_params.stimulus,
        lattice = base_params.lattice,
        pop_names = base_params.pop_names
    )
end

"""
    create_oscillating_pulse_parameters(; lattice=nothing, kwargs...)

Create parameters for oscillating pulse dynamics in 1D.

This parameterization is designed to produce oscillatory pulses that can
propagate along a 1D neural field.

# Keyword Arguments
- `lattice`: Spatial lattice (defaults to 1D periodic lattice if not provided)
- `Aee`, `See`: Excitatory-to-excitatory amplitude and spread (default: 16.0, 2.5)
- `Aii`, `Sii`: Inhibitory-to-inhibitory amplitude and spread (default: 4.0, 2.7)
- `Aie`, `Sie`: Excitatory-to-inhibitory amplitude and spread (default: 27.0, 2.5)
- `Aei`, `Sei`: Inhibitory-to-excitatory amplitude and spread (default: 18.2, 2.7)
- `aE`, `θE`: Excitatory nonlinearity slope and threshold (default: 1.2, 2.6)
- `aI`, `θI`: Inhibitory nonlinearity slope and threshold (default: 1.0, 8.0)
- `α`: Decay rates for (E, I) (default: (1.5, 1.0))
- `β`: Saturation coefficients for (E, I) (default: (1.1, 1.1))
- `τ`: Time constants for (E, I) (default: (10.0, 18.0))

# Returns
WilsonCowanParameters{2} configured for oscillating pulse dynamics
"""
function create_oscillating_pulse_parameters(;
    lattice=nothing,
    Aee=16.0, See=2.5,
    Aii=4.0, Sii=2.7,
    Aie=27.0, Sie=2.5,
    Aei=18.2, Sei=2.7,
    aE=1.2, θE=2.6,
    aI=1.0, θI=8.0,
    α=(1.5, 1.0),
    β=(1.1, 1.1),
    τ=(10.0, 18.0))
    
    # Create default lattice if not provided
    if lattice === nothing
        lattice = PeriodicLattice(extent=(100.0,), n_points=(100,))
    end
    
    # Create nonlinearity for each population
    # Use rectified nonlinearities for biological realism
    nonlinearity_e = RectifiedZeroedSigmoidNonlinearity(a=aE, θ=θE)
    nonlinearity_i = RectifiedZeroedSigmoidNonlinearity(a=aI, θ=θI)
    nonlinearity = (nonlinearity_e, nonlinearity_i)
    
    # Create connectivity matrix
    conn_ee = GaussianConnectivityParameter(Aee, (See,))
    conn_ei = GaussianConnectivityParameter(-Aei, (Sei,))
    conn_ie = GaussianConnectivityParameter(Aie, (Sie,))
    conn_ii = GaussianConnectivityParameter(-Aii, (Sii,))
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    # Create parameters
    params = WilsonCowanParameters{2}(
        α = α,
        β = β,
        τ = τ,
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    return params
end

"""
    create_propagating_torus_parameters(; lattice=nothing, kwargs...)

Create parameters for propagating patterns on 2D torus (periodic lattice).

This parameterization is designed to produce propagating wave patterns
on a 2D periodic lattice (torus topology).

# Keyword Arguments
- `lattice`: Spatial lattice (defaults to 2D periodic lattice if not provided)
- Connectivity and nonlinearity parameters similar to `create_oscillating_pulse_parameters`
- `α`: Decay rates for (E, I) (default: (1.0, 1.0))
- `β`: Saturation coefficients for (E, I) (default: (1.0, 1.0))
- `τ`: Time constants for (E, I) (default: (3.0, 3.0))

# Returns
WilsonCowanParameters{2} configured for propagating patterns on torus
"""
function create_propagating_torus_parameters(;
    lattice=nothing,
    Aee=16.0, See=2.5,
    Aii=4.0, Sii=2.7,
    Aie=27.0, Sie=2.5,
    Aei=18.2, Sei=2.7,
    aE=1.2, θE=2.6,
    aI=1.0, θI=8.0,
    α=(1.0, 1.0),
    β=(1.0, 1.0),
    τ=(3.0, 3.0))
    
    # Create default lattice if not provided - 2D torus
    if lattice === nothing
        lattice = PeriodicLattice(extent=(100.0, 100.0), n_points=(100, 100))
    end
    
    # Create nonlinearity for each population
    nonlinearity_e = RectifiedZeroedSigmoidNonlinearity(a=aE, θ=θE)
    nonlinearity_i = RectifiedZeroedSigmoidNonlinearity(a=aI, θ=θI)
    nonlinearity = (nonlinearity_e, nonlinearity_i)
    
    # Create connectivity matrix - 2D spreads
    conn_ee = GaussianConnectivityParameter(Aee, (See, See))
    conn_ei = GaussianConnectivityParameter(-Aei, (Sei, Sei))
    conn_ie = GaussianConnectivityParameter(Aie, (Sie, Sie))
    conn_ii = GaussianConnectivityParameter(-Aii, (Sii, Sii))
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    # Create parameters
    params = WilsonCowanParameters{2}(
        α = α,
        β = β,
        τ = τ,
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    return params
end
