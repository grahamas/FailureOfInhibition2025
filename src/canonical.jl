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
