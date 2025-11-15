"""
Parameter optimization for traveling waves in 1D Wilson-Cowan models.

This module provides functions to optimize model parameters to achieve
desired traveling wave behaviors, using the traveling wave metrics from analysis.jl.
"""

using Optim
using Statistics

"""
    TravelingWaveObjective

Structure defining the objective for traveling wave optimization.

# Fields
- `target_distance::Union{Float64, Nothing}`: Target distance the wave should travel (nothing = maximize)
- `target_amplitude::Union{Float64, Nothing}`: Target maximum amplitude (nothing = unconstrained)
- `target_width::Union{Float64, Nothing}`: Target spatial width (nothing = unconstrained)
- `minimize_decay::Bool`: Whether to minimize decay rate (default: true)
- `require_traveling::Bool`: Whether to require detection of a traveling peak (default: true)
- `threshold::Float64`: Threshold for peak detection (default: 0.15)
"""
struct TravelingWaveObjective
    target_distance::Union{Float64, Nothing}
    target_amplitude::Union{Float64, Nothing}
    target_width::Union{Float64, Nothing}
    minimize_decay::Bool
    require_traveling::Bool
    threshold::Float64
    
    function TravelingWaveObjective(;
        target_distance=nothing,
        target_amplitude=nothing,
        target_width=nothing,
        minimize_decay=true,
        require_traveling=true,
        threshold=0.15
    )
        new(target_distance, target_amplitude, target_width, 
            minimize_decay, require_traveling, threshold)
    end
end

"""
    optimize_for_traveling_wave(
        base_params::WilsonCowanParameters{P},
        param_ranges::NamedTuple,
        objective::TravelingWaveObjective,
        A₀,
        tspan;
        saveat=0.2,
        method=BFGS(),
        maxiter=100
    ) where P

Optimize Wilson-Cowan parameters to achieve desired traveling wave behavior.

# Arguments
- `base_params`: Base parameter set to start from
- `param_ranges`: Named tuple of (min, max) ranges for parameters to optimize
  Example: `(connectivity_width = (1.0, 5.0), sigmoid_a = (1.0, 4.0))`
- `objective`: TravelingWaveObjective specifying the optimization goal
- `A₀`: Initial condition for simulations
- `tspan`: Time span for simulations
- `saveat`: Time step for saving simulation results (default: 0.2)
- `method`: Optimization method from Optim.jl (default: BFGS())
- `maxiter`: Maximum number of iterations (default: 100)

# Returns
- `result`: Optim.jl optimization result
- `best_params`: WilsonCowanParameters with optimized values

# Examples
```julia
# Set up base parameters
lattice = CompactLattice(extent=(20.0,), n_points=(101,))
conn = GaussianConnectivityParameter(1.0, (2.0,))
params = WilsonCowanParameters{1}(
    α=(1.0,), β=(1.0,), τ=(8.0,),
    connectivity=ConnectivityMatrix{1}(reshape([conn], 1, 1)),
    nonlinearity=SigmoidNonlinearity(a=2.0, θ=0.25),
    stimulus=nothing, lattice=lattice, pop_names=("E",)
)

# Define what to optimize
param_ranges = (
    connectivity_width = (1.5, 4.0),
    sigmoid_a = (1.5, 3.5)
)

# Define objective
objective = TravelingWaveObjective(
    target_distance=10.0,
    minimize_decay=true,
    require_traveling=true
)

# Initial condition
A₀ = zeros(101, 1)
A₀[15:20, 1] .= 0.6

# Optimize
result, best_params = optimize_for_traveling_wave(
    params, param_ranges, objective, A₀, (0.0, 40.0)
)
```
"""
function optimize_for_traveling_wave(
    base_params::WilsonCowanParameters{P},
    param_ranges::NamedTuple,
    objective::TravelingWaveObjective,
    A₀,
    tspan;
    saveat=0.2,
    method=BFGS(),
    maxiter=100
) where P
    # Extract parameter names and bounds
    param_names = keys(param_ranges)
    lower_bounds = [param_ranges[k][1] for k in param_names]
    upper_bounds = [param_ranges[k][2] for k in param_names]
    
    # Initial guess: middle of the range
    x0 = [(lower_bounds[i] + upper_bounds[i]) / 2 for i in 1:length(param_names)]
    
    # Define objective function
    function obj_function(x)
        # Create parameters with current values
        current_params = _update_params(base_params, param_names, x)
        
        # Run simulation
        try
            sol = solve_model(A₀, tspan, current_params, saveat=saveat)
            
            # Compute metrics
            has_peak, _, _ = detect_traveling_peak(sol, 1, threshold=objective.threshold)
            
            # If we require traveling and don't have it, return high penalty
            if objective.require_traveling && !has_peak
                return 1e6
            end
            
            # Compute distance
            distance, _ = compute_distance_traveled(sol, 1, current_params.lattice, threshold=objective.threshold)
            
            # Compute amplitude
            amplitude = compute_amplitude(sol, 1, method=:max)
            
            # Compute width
            width, _, _ = compute_half_max_width(sol, 1, nothing, current_params.lattice)
            
            # Compute decay rate
            decay_rate, _ = compute_decay_rate(sol, 1)
            
            # Calculate loss
            loss = 0.0
            
            # Distance loss
            if objective.target_distance !== nothing
                loss += (distance - objective.target_distance)^2
            else
                # Maximize distance (minimize negative distance)
                loss -= distance
            end
            
            # Amplitude loss
            if objective.target_amplitude !== nothing
                loss += 10.0 * (amplitude - objective.target_amplitude)^2
            end
            
            # Width loss
            if objective.target_width !== nothing
                loss += 5.0 * (width - objective.target_width)^2
            end
            
            # Decay loss
            if objective.minimize_decay && decay_rate !== nothing
                loss += decay_rate * 100.0  # Penalize decay
            end
            
            return loss
        catch e
            # If simulation fails, return high penalty
            @warn "Simulation failed with parameters: $x" exception=e
            return 1e6
        end
    end
    
    # Run optimization with bounds
    result = optimize(
        obj_function,
        lower_bounds,
        upper_bounds,
        x0,
        Fminbox(method),
        Optim.Options(iterations=maxiter, show_trace=false)
    )
    
    # Create best parameters
    best_x = Optim.minimizer(result)
    best_params = _update_params(base_params, param_names, best_x)
    
    return result, best_params
end

"""
    _update_params(base_params, param_names, values)

Helper function to update parameters based on optimization variables.

Supports updating:
- `connectivity_width`: Spread (standard deviation) of Gaussian connectivity (for 1D, single population)
- `connectivity_strength`: Amplitude of Gaussian connectivity
- `sigmoid_a`: Sigmoid steepness parameter
- `sigmoid_θ`: Sigmoid threshold parameter
- `τ`: Time constant (for single population)
"""
function _update_params(base_params::WilsonCowanParameters, param_names, values)
    # Extract type parameters
    T = eltype(base_params.α)
    P = length(base_params.α)
    
    # Create a mutable copy of parameters
    params_dict = Dict{Symbol, Any}()
    params_dict[:α] = base_params.α
    params_dict[:β] = base_params.β
    params_dict[:τ] = base_params.τ
    params_dict[:connectivity] = base_params.connectivity
    params_dict[:nonlinearity] = base_params.nonlinearity
    params_dict[:stimulus] = base_params.stimulus
    params_dict[:lattice] = base_params.lattice
    params_dict[:pop_names] = base_params.pop_names
    
    # Update connectivity if specified
    if :connectivity_width in param_names || :connectivity_strength in param_names
        # Extract current connectivity (assuming Gaussian for now)
        conn_matrix = base_params.connectivity
        if P == 1
            # Single population
            old_conn = conn_matrix[1, 1]
            if old_conn isa GaussianConnectivityParameter
                new_spread = old_conn.spread
                new_amplitude = old_conn.amplitude
                
                for (i, name) in enumerate(param_names)
                    if name == :connectivity_width
                        new_spread = (values[i],)
                    elseif name == :connectivity_strength
                        new_amplitude = values[i]
                    end
                end
                
                new_conn = GaussianConnectivityParameter(new_amplitude, new_spread)
                params_dict[:connectivity] = ConnectivityMatrix{1}(reshape([new_conn], 1, 1))
            elseif old_conn isa GaussianConnectivity
                # Handle pre-computed connectivity
                new_spread = old_conn.spread
                new_amplitude = old_conn.amplitude
                
                for (i, name) in enumerate(param_names)
                    if name == :connectivity_width
                        new_spread = (values[i],)
                    elseif name == :connectivity_strength
                        new_amplitude = values[i]
                    end
                end
                
                new_conn = GaussianConnectivityParameter(new_amplitude, new_spread)
                params_dict[:connectivity] = ConnectivityMatrix{1}(reshape([new_conn], 1, 1))
            end
        end
    end
    
    # Update nonlinearity if specified
    if :sigmoid_a in param_names || :sigmoid_θ in param_names
        old_nonlin = base_params.nonlinearity
        if old_nonlin isa SigmoidNonlinearity
            new_a = old_nonlin.a
            new_θ = old_nonlin.θ
            
            for (i, name) in enumerate(param_names)
                if name == :sigmoid_a
                    new_a = values[i]
                elseif name == :sigmoid_θ
                    new_θ = values[i]
                end
            end
            
            params_dict[:nonlinearity] = SigmoidNonlinearity(a=new_a, θ=new_θ)
        end
    end
    
    # Update time constant if specified
    if :τ in param_names && P == 1
        for (i, name) in enumerate(param_names)
            if name == :τ
                params_dict[:τ] = (values[i],)
            end
        end
    end
    
    # Create new parameters using keyword constructor to ensure connectivity is prepared
    return WilsonCowanParameters{P}(
        α = params_dict[:α],
        β = params_dict[:β],
        τ = params_dict[:τ],
        connectivity = params_dict[:connectivity],
        nonlinearity = params_dict[:nonlinearity],
        stimulus = params_dict[:stimulus],
        lattice = params_dict[:lattice],
        pop_names = params_dict[:pop_names]
    )
end
