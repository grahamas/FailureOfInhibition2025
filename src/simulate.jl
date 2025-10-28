"""
Simulation utilities for Wilson-Cowan models using DifferentialEquations.jl
"""

using DifferentialEquations
using DataFrames
using CSV
using Statistics

"""
    solve_model(initial_condition, tspan, params::WilsonCowanParameters{N}; solver=Tsit5(), kwargs...) where N

Solve the Wilson-Cowan model using DifferentialEquations.jl.

# Arguments
- `initial_condition`: Initial state array (shape depends on model type)
  - Point models: `(1, P)` array for P populations with connectivity
  - Point models without connectivity: `[A₁, ..., Aₚ]` vector
  - Spatial models: `(N_points, P)` array for N_points spatial locations and P populations
- `tspan`: Tuple `(t_start, t_end)` specifying the simulation time span
- `params`: WilsonCowanParameters containing model configuration

# Keyword Arguments
- `solver`: ODE solver algorithm (default: Tsit5())
- `kwargs...`: Additional keyword arguments passed to `solve()`
  - `saveat`: Times at which to save the solution (default: automatic)
  - `reltol`: Relative tolerance (default: 1e-6)
  - `abstol`: Absolute tolerance (default: 1e-8)
  - `maxiters`: Maximum number of iterations (default: 1e5)

# Returns
- ODE solution object from DifferentialEquations.jl
  - Access times with `sol.t`
  - Access states with `sol.u` (array of states at each time)
  - Can be indexed like `sol[i]` for state at time i
  - Can be called like `sol(t)` for interpolated state at time t

# Examples

## Point model (2 populations with connectivity)
```julia
using FailureOfInhibition2025

lattice = PointLattice()
connectivity = ConnectivityMatrix{2}([
    ScalarConnectivity(1.0) ScalarConnectivity(-0.5);
    ScalarConnectivity(0.8) ScalarConnectivity(-0.3)
])

params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (1.0, 0.8),
    connectivity = connectivity,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

A₀ = reshape([0.1, 0.1], 1, 2)
tspan = (0.0, 100.0)
sol = solve_model(A₀, tspan, params)
```

## Spatial model (1D with 2 populations)
```julia
lattice = CompactLattice(extent=(10.0,), n_points=(101,))
connectivity = ConnectivityMatrix{2}([
    GaussianConnectivityParameter(1.0, (2.0,)) GaussianConnectivityParameter(-0.5, (1.5,));
    GaussianConnectivityParameter(0.8, (2.5,)) GaussianConnectivityParameter(-0.3, (1.0,))
])

params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (1.0, 0.8),
    connectivity = connectivity,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
    stimulus = CircleStimulus(radius=2.0, strength=0.5, time_windows=[(0.0, 10.0)], lattice=lattice),
    lattice = lattice,
    pop_names = ("E", "I")
)

A₀ = 0.1 .+ 0.05 .* rand(101, 2)
tspan = (0.0, 100.0)
sol = solve_model(A₀, tspan, params, saveat=0.1)
```
"""
function solve_model(initial_condition, tspan, params::WilsonCowanParameters{N}; solver=Tsit5(), kwargs...) where N
    # Create ODE problem
    prob = ODEProblem(wcm1973!, initial_condition, tspan, params)
    
    # Solve the problem
    sol = solve(prob, solver; kwargs...)
    
    return sol
end

"""
    save_simulation_results(sol, filename; params=nothing)

Save complete simulation results to a CSV file.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `filename`: Output CSV file path
- `params`: (Optional) WilsonCowanParameters to include metadata

# File Format
The CSV file contains columns:
- `time`: Time points
- `pop1`, `pop2`, ...: Activity values for each population
  - For spatial models, columns are `pop1_point1`, `pop1_point2`, etc.

# Examples

```julia
sol = solve_model(A₀, tspan, params)
save_simulation_results(sol, "results.csv", params=params)
```
"""
function save_simulation_results(sol, filename; params=nothing)
    # Extract dimensions from the solution
    first_state = sol.u[1]
    
    # Determine if this is a spatial or point model
    if ndims(first_state) == 1
        # Simple 1D array (point model without connectivity)
        n_pops = length(first_state)
        is_spatial = false
        n_points = 1
    elseif ndims(first_state) == 2
        # 2D array (spatial model or point model with connectivity)
        n_points, n_pops = size(first_state)
        is_spatial = (n_points > 1)
    else
        error("Unexpected state dimensionality: $(ndims(first_state))")
    end
    
    # Get population names if available
    if params !== nothing && hasfield(typeof(params), :pop_names)
        pop_names = params.pop_names
    else
        pop_names = ntuple(i -> "pop$i", n_pops)
    end
    
    # Create DataFrame
    df = DataFrame()
    df.time = sol.t
    
    # Add population data
    if is_spatial
        # Spatial model: add columns for each population at each point
        for pop_idx in 1:n_pops
            for point_idx in 1:n_points
                col_name = Symbol("$(pop_names[pop_idx])_point$(point_idx)")
                df[!, col_name] = [u[point_idx, pop_idx] for u in sol.u]
            end
        end
    else
        # Point model: simple columns for each population
        for pop_idx in 1:n_pops
            col_name = Symbol(pop_names[pop_idx])
            if ndims(first_state) == 1
                df[!, col_name] = [u[pop_idx] for u in sol.u]
            else
                df[!, col_name] = [u[1, pop_idx] for u in sol.u]
            end
        end
    end
    
    # Write to CSV
    CSV.write(filename, df)
    
    return df
end

"""
    save_simulation_summary(sol, filename; params=nothing)

Save simulation summary statistics to a CSV file.

# Arguments
- `sol`: ODE solution object from `solve_model()`
- `filename`: Output CSV file path
- `params`: (Optional) WilsonCowanParameters to include metadata

# File Format
The CSV file contains summary statistics for each population:
- `population`: Population name
- `mean`: Mean activity over time
- `std`: Standard deviation of activity
- `min`: Minimum activity
- `max`: Maximum activity
- `final`: Final activity value
- For spatial models, also includes spatial statistics (spatial_mean, spatial_std)

# Examples

```julia
sol = solve_model(A₀, tspan, params)
save_simulation_summary(sol, "summary.csv", params=params)
```
"""
function save_simulation_summary(sol, filename; params=nothing)
    # Extract dimensions from the solution
    first_state = sol.u[1]
    
    # Determine if this is a spatial or point model
    if ndims(first_state) == 1
        # Simple 1D array (point model without connectivity)
        n_pops = length(first_state)
        is_spatial = false
        n_points = 1
    elseif ndims(first_state) == 2
        # 2D array (spatial model or point model with connectivity)
        n_points, n_pops = size(first_state)
        is_spatial = (n_points > 1)
    else
        error("Unexpected state dimensionality: $(ndims(first_state))")
    end
    
    # Get population names if available
    if params !== nothing && hasfield(typeof(params), :pop_names)
        pop_names = params.pop_names
    else
        pop_names = ntuple(i -> "pop$i", n_pops)
    end
    
    # Create summary DataFrame
    summary_data = []
    
    for pop_idx in 1:n_pops
        # Extract time series for this population
        if ndims(first_state) == 1
            time_series = [u[pop_idx] for u in sol.u]
        elseif is_spatial
            # For spatial models, collect all spatial points at all times
            time_series = reduce(vcat, [u[:, pop_idx] for u in sol.u])
        else
            # Point model with connectivity (1, P) shape
            time_series = [u[1, pop_idx] for u in sol.u]
        end
        
        # Calculate statistics
        pop_stats = Dict(
            "population" => string(pop_names[pop_idx]),
            "mean" => mean(time_series),
            "std" => std(time_series),
            "min" => minimum(time_series),
            "max" => maximum(time_series),
            "final" => is_spatial ? mean(sol.u[end][:, pop_idx]) : (ndims(first_state) == 1 ? sol.u[end][pop_idx] : sol.u[end][1, pop_idx])
        )
        
        # Add spatial statistics if applicable
        if is_spatial
            # Calculate spatial variability at each time point
            spatial_stds = [std(u[:, pop_idx]) for u in sol.u]
            spatial_means = [mean(u[:, pop_idx]) for u in sol.u]
            
            pop_stats["spatial_mean_of_means"] = mean(spatial_means)
            pop_stats["spatial_mean_of_stds"] = mean(spatial_stds)
            pop_stats["final_spatial_mean"] = mean(sol.u[end][:, pop_idx])
            pop_stats["final_spatial_std"] = std(sol.u[end][:, pop_idx])
        end
        
        push!(summary_data, pop_stats)
    end
    
    # Create DataFrame from summary data
    df = DataFrame(summary_data)
    
    # Write to CSV
    CSV.write(filename, df)
    
    return df
end
