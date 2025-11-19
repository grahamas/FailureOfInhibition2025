module FailureOfInhibition2025CUDAExt

using FailureOfInhibition2025
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using Statistics
using LinearAlgebra

"""
    FailureOfInhibition2025.propagate_activation_single(dA::CuArray, A::CuArray, connectivity::FailureOfInhibition2025.GaussianConnectivityParameter, t, lattice)

GPU-specific version of propagate_activation_single that creates a GaussianConnectivity object
with CUDA FFT plans for GPU arrays.
"""
function FailureOfInhibition2025.propagate_activation_single(
    dA::CuArray, 
    A::CuArray, 
    connectivity::FailureOfInhibition2025.GaussianConnectivityParameter, 
    t, 
    lattice
)
    # Create GPU-compatible GaussianConnectivity
    gc = create_gpu_gaussian_connectivity(connectivity, lattice, A)
    FailureOfInhibition2025.propagate_activation_single(dA, A, gc, t, lattice)
end

"""
    create_gpu_gaussian_connectivity(param::GaussianConnectivityParameter, lattice, array_template::CuArray)

Helper function to create a GaussianConnectivity object with CUDA FFT plans.
"""
function create_gpu_gaussian_connectivity(
    param::FailureOfInhibition2025.GaussianConnectivityParameter, 
    lattice,
    array_template::CuArray
)
    # Calculate kernel on CPU first, then transfer to GPU
    kernel_cpu = FailureOfInhibition2025.calculate_kernel(param, lattice)
    kernel = CuArray(kernel_cpu)
    
    # Create GPU FFT plans
    fft_op = plan_rfft(kernel)
    kernel_fft = fft_op * kernel
    ifft_op = plan_irfft(kernel_fft, size(kernel, 1))
    
    # Create GPU buffers
    buffer_real = similar(kernel)
    buffer_complex = similar(kernel_fft)
    buffer_shift = similar(buffer_real)
    
    return FailureOfInhibition2025.GaussianConnectivity(
        fft_op, ifft_op, kernel_fft, buffer_real, buffer_complex, buffer_shift
    )
end

"""
    solve_model_gpu(initial_condition, tspan, params::WilsonCowanParameters{N}; solver=Tsit5(), kwargs...) where N

GPU-accelerated version of solve_model using CUDA.jl.

Automatically transfers data to GPU, runs the simulation, and transfers results back to CPU.

# Arguments
- `initial_condition`: Initial state array (will be transferred to GPU)
- `tspan`: Tuple `(t_start, t_end)` specifying the simulation time span
- `params`: WilsonCowanParameters containing model configuration

# Keyword Arguments
- `solver`: ODE solver algorithm (default: Tsit5())
- `kwargs...`: Additional keyword arguments passed to `solve()`

# Returns
- ODE solution object (on CPU) from DifferentialEquations.jl

# Examples

```julia
using FailureOfInhibition2025
using CUDA

# Ensure CUDA is available
if CUDA.functional()
    lattice = CompactLattice(extent=(10.0,), n_points=(101,))
    connectivity = ConnectivityMatrix{1}([
        GaussianConnectivityParameter(1.0, (2.0,))
    ])
    
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (8.0,),
        connectivity = connectivity,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E",)
    )
    
    A₀ = zeros(101, 1)
    A₀[15:20, 1] .= 0.6
    tspan = (0.0, 40.0)
    
    # Run on GPU
    sol = solve_model_gpu(A₀, tspan, params, saveat=0.1)
end
```
"""
function FailureOfInhibition2025.solve_model_gpu(initial_condition, tspan, params::FailureOfInhibition2025.WilsonCowanParameters{N}; solver=Tsit5(), kwargs...) where N
    # Transfer initial condition to GPU
    u0_gpu = CuArray(initial_condition)
    
    # Create ODE problem on GPU
    prob = ODEProblem(FailureOfInhibition2025.wcm1973!, u0_gpu, tspan, params)
    
    # Solve the problem on GPU
    sol = solve(prob, solver; kwargs...)
    
    # Transfer solution back to CPU
    sol_cpu = DiffEqBase.DESolution(
        [Array(u) for u in sol.u],
        sol.t,
        sol.prob,
        sol.alg,
        sol.dense,
        sol.tslocation,
        sol.stats,
        sol.alg_choice,
        sol.retcode,
        sol.destats,
        sol.interpolation
    )
    
    return sol_cpu
end

"""
    create_gpu_output_function(tspan, output_metric::Symbol, saveat=nothing)

Create a GPU-enabled output function for sensitivity analysis.

# Arguments
- `tspan`: Time span for simulation
- `output_metric`: Symbol specifying which metric to compute
- `saveat`: Time points to save (default: adaptive)

# Returns
A function `f(params) -> scalar` that simulates on GPU and returns the metric.
"""
function create_gpu_output_function(tspan, output_metric::Symbol, saveat=nothing)
    function compute_output(params::FailureOfInhibition2025.WilsonCowanParameters{T,P}) where {T,P}
        # Create initial condition
        if params.lattice isa FailureOfInhibition2025.PointLattice
            A₀ = reshape(fill(0.1, P), 1, P)
        else
            n_points = size(FailureOfInhibition2025.coordinates(params.lattice), 1)
            A₀ = fill(0.1, n_points, P)
        end
        
        # Solve model on GPU
        try
            if saveat === nothing
                sol = FailureOfInhibition2025.solve_model_gpu(A₀, tspan, params)
            else
                sol = FailureOfInhibition2025.solve_model_gpu(A₀, tspan, params, saveat=saveat)
            end
            
            # Compute requested metric (same logic as CPU version)
            if output_metric == :final_mean
                final_state = sol.u[end]
                return mean(final_state)
                
            elseif output_metric == :final_E
                final_state = sol.u[end]
                if ndims(final_state) == 2
                    return mean(final_state[:, 1])
                else
                    return final_state[1]
                end
                
            elseif output_metric == :final_I && P >= 2
                final_state = sol.u[end]
                if ndims(final_state) == 2
                    return mean(final_state[:, 2])
                else
                    return final_state[2]
                end
                
            elseif output_metric == :max_mean
                max_activity = 0.0
                for u in sol.u
                    max_activity = max(max_activity, mean(u))
                end
                return max_activity
                
            elseif output_metric == :variance
                mean_activities = [mean(u) for u in sol.u]
                return var(mean_activities)
                
            elseif output_metric == :oscillation_amplitude
                n_points = length(sol.u)
                start_idx = max(1, div(4 * n_points, 5))
                final_activities = [mean(u) for u in sol.u[start_idx:end]]
                return maximum(final_activities) - minimum(final_activities)
                
            else
                error("Unknown output metric: $output_metric")
            end
        catch e
            @warn "GPU simulation failed: $e"
            return NaN
        end
    end
    
    return compute_output
end

"""
    sobol_sensitivity_analysis_gpu(
        base_params::WilsonCowanParameters,
        param_ranges,
        n_samples=1000;
        tspan=(0.0, 100.0),
        output_metric=:final_mean,
        saveat=nothing
    )

GPU-accelerated Sobol global sensitivity analysis.

Runs simulations on GPU for faster parameter exploration. Automatically falls back
to CPU if GPU is not available.

# Arguments
- `base_params`: Base WilsonCowanParameters to use as template
- `param_ranges`: Vector of tuples (param_name, lower_bound, upper_bound)
- `n_samples`: Number of samples for Monte Carlo estimation (default: 1000)
- `tspan`: Time span for simulation (default: (0.0, 100.0))
- `output_metric`: Which output metric to analyze (default: :final_mean)
- `saveat`: Time points to save (default: adaptive)

# Returns
A dictionary containing:
- `:S1`: First-order Sobol indices
- `:ST`: Total-order Sobol indices
- `:param_names`: Names of parameters analyzed
- `:method`: "Sobol (GPU)"

# Example

```julia
using FailureOfInhibition2025
using CUDA

if CUDA.functional()
    result = sobol_sensitivity_analysis_gpu(
        base_params,
        [("α_E", 0.5, 2.0), ("α_I", 0.5, 2.0)],
        n_samples=500
    )
    println("First-order indices: ", result[:S1])
end
```
"""
function FailureOfInhibition2025.sobol_sensitivity_analysis_gpu(
    base_params::FailureOfInhibition2025.WilsonCowanParameters,
    param_ranges,
    n_samples=1000;
    tspan=(0.0, 100.0),
    output_metric=:final_mean,
    saveat=nothing
)
    # Check if CUDA is available
    if !CUDA.functional()
        @warn "CUDA not available, falling back to CPU version"
        return FailureOfInhibition2025.sobol_sensitivity_analysis(
            base_params, param_ranges, n_samples;
            tspan=tspan, output_metric=output_metric, saveat=saveat
        )
    end
    
    # Extract bounds
    n_params = length(param_ranges)
    param_names = [pr[1] for pr in param_ranges]
    lb = [pr[2] for pr in param_ranges]
    ub = [pr[3] for pr in param_ranges]
    
    # Create parameter builder and GPU output function
    build_params = FailureOfInhibition2025.create_parameter_builder(base_params, param_ranges)
    compute_output = create_gpu_output_function(tspan, output_metric, saveat)
    
    # Create wrapper function for GlobalSensitivity.jl
    function f(p_matrix)
        n_samples_run = size(p_matrix, 2)
        outputs = zeros(n_samples_run)
        
        for i in 1:n_samples_run
            p_vector = p_matrix[:, i]
            params = build_params(p_vector)
            outputs[i] = compute_output(params)
        end
        
        return outputs
    end
    
    # Run Sobol analysis
    println("Running Sobol sensitivity analysis on GPU with $n_samples samples...")
    println("Parameters: ", param_names)
    println("Output metric: $output_metric")
    
    result = GlobalSensitivity.gsa(f, GlobalSensitivity.Sobol(), [[l, u] for (l, u) in zip(lb, ub)], samples=n_samples)
    
    return Dict(
        :S1 => result.S1,
        :ST => result.ST,
        :param_names => param_names,
        :method => "Sobol (GPU)",
        :n_samples => n_samples,
        :output_metric => output_metric
    )
end

"""
    morris_sensitivity_analysis_gpu(
        base_params::WilsonCowanParameters,
        param_ranges,
        n_trajectories=100;
        tspan=(0.0, 100.0),
        output_metric=:final_mean,
        saveat=nothing,
        num_levels=10
    )

GPU-accelerated Morris screening sensitivity analysis.

# Arguments
- `base_params`: Base WilsonCowanParameters to use as template
- `param_ranges`: Vector of tuples (param_name, lower_bound, upper_bound)
- `n_trajectories`: Number of trajectories to sample (default: 100)
- `tspan`: Time span for simulation (default: (0.0, 100.0))
- `output_metric`: Which output metric to analyze (default: :final_mean)
- `saveat`: Time points to save (default: adaptive)
- `num_levels`: Number of grid levels for Morris sampling (default: 10)

# Returns
A dictionary containing:
- `:means`: Mean of elementary effects
- `:means_star`: Mean of absolute elementary effects
- `:variances`: Variance of elementary effects
- `:param_names`: Names of parameters analyzed
- `:method`: "Morris (GPU)"

# Example

```julia
using FailureOfInhibition2025
using CUDA

if CUDA.functional()
    result = morris_sensitivity_analysis_gpu(
        base_params,
        [("α_E", 0.5, 2.0), ("α_I", 0.5, 2.0)],
        n_trajectories=50
    )
    println("Mean effects: ", result[:means_star])
end
```
"""
function FailureOfInhibition2025.morris_sensitivity_analysis_gpu(
    base_params::FailureOfInhibition2025.WilsonCowanParameters,
    param_ranges,
    n_trajectories=100;
    tspan=(0.0, 100.0),
    output_metric=:final_mean,
    saveat=nothing,
    num_levels=10
)
    # Check if CUDA is available
    if !CUDA.functional()
        @warn "CUDA not available, falling back to CPU version"
        return FailureOfInhibition2025.morris_sensitivity_analysis(
            base_params, param_ranges, n_trajectories;
            tspan=tspan, output_metric=output_metric, saveat=saveat, num_levels=num_levels
        )
    end
    
    # Extract bounds
    n_params = length(param_ranges)
    param_names = [pr[1] for pr in param_ranges]
    lb = [pr[2] for pr in param_ranges]
    ub = [pr[3] for pr in param_ranges]
    
    # Create parameter builder and GPU output function
    build_params = FailureOfInhibition2025.create_parameter_builder(base_params, param_ranges)
    compute_output = create_gpu_output_function(tspan, output_metric, saveat)
    
    # Create wrapper function for GlobalSensitivity.jl
    function f(p_matrix)
        n_samples = size(p_matrix, 2)
        outputs = zeros(n_samples)
        
        for i in 1:n_samples
            p_vector = p_matrix[:, i]
            params = build_params(p_vector)
            outputs[i] = compute_output(params)
        end
        
        return outputs
    end
    
    # Run Morris analysis
    println("Running Morris sensitivity analysis on GPU with $n_trajectories trajectories...")
    println("Parameters: ", param_names)
    println("Output metric: $output_metric")
    
    result = GlobalSensitivity.gsa(f, GlobalSensitivity.Morris(), [[l, u] for (l, u) in zip(lb, ub)], 
                 num_trajectory=n_trajectories, total_num_trajectory=n_trajectories,
                 num_levels=num_levels)
    
    return Dict(
        :means => result.means,
        :means_star => result.means_star,
        :variances => result.variances,
        :param_names => param_names,
        :method => "Morris (GPU)",
        :n_trajectories => n_trajectories,
        :output_metric => output_metric
    )
end

"""
    optimize_for_traveling_wave_gpu(
        base_params::WilsonCowanParameters{P},
        param_ranges::NamedTuple,
        objective::TravelingWaveObjective,
        A₀,
        tspan;
        saveat=0.2,
        method=BFGS(),
        maxiter=100
    ) where P

GPU-accelerated optimization for traveling wave parameters.

# Arguments
- `base_params`: Base parameter set to start from
- `param_ranges`: Named tuple of (min, max) ranges for parameters to optimize
- `objective`: TravelingWaveObjective specifying the optimization goal
- `A₀`: Initial condition for simulations
- `tspan`: Time span for simulations
- `saveat`: Time step for saving simulation results (default: 0.2)
- `method`: Optimization method from Optim.jl (default: BFGS())
- `maxiter`: Maximum number of iterations (default: 100)

# Returns
- `result`: Optim.jl optimization result
- `best_params`: WilsonCowanParameters with optimized values

# Example

```julia
using FailureOfInhibition2025
using CUDA

if CUDA.functional()
    result, best_params = optimize_for_traveling_wave_gpu(
        params, param_ranges, objective, A₀, (0.0, 40.0)
    )
end
```
"""
function FailureOfInhibition2025.optimize_for_traveling_wave_gpu(
    base_params::FailureOfInhibition2025.WilsonCowanParameters{P},
    param_ranges::NamedTuple,
    objective::FailureOfInhibition2025.TravelingWaveObjective,
    A₀,
    tspan;
    saveat=0.2,
    method=Optim.BFGS(),
    maxiter=100
) where P
    # Check if CUDA is available
    if !CUDA.functional()
        @warn "CUDA not available, falling back to CPU version"
        return FailureOfInhibition2025.optimize_for_traveling_wave(
            base_params, param_ranges, objective, A₀, tspan;
            saveat=saveat, method=method, maxiter=maxiter
        )
    end
    
    # Extract parameter names and bounds
    param_names = keys(param_ranges)
    lower_bounds = [param_ranges[k][1] for k in param_names]
    upper_bounds = [param_ranges[k][2] for k in param_names]
    
    # Initial guess: middle of the range
    x0 = [(lower_bounds[i] + upper_bounds[i]) / 2 for i in 1:length(param_names)]
    
    # Define objective function (runs simulations on GPU)
    function obj_function(x)
        # Create parameters with current values
        current_params = FailureOfInhibition2025._update_params(base_params, param_names, x)
        
        # Run simulation on GPU
        try
            sol = FailureOfInhibition2025.solve_model_gpu(A₀, tspan, current_params, saveat=saveat)
            
            # Compute metrics
            has_peak, _, _ = FailureOfInhibition2025.detect_traveling_peak(sol, 1, threshold=objective.threshold)
            
            # If we require traveling and don't have it, return high penalty
            if objective.require_traveling && !has_peak
                return 1e6
            end
            
            # Compute distance
            distance, _ = FailureOfInhibition2025.compute_distance_traveled(sol, 1, current_params.lattice, threshold=objective.threshold)
            
            # Compute amplitude
            amplitude = FailureOfInhibition2025.compute_amplitude(sol, 1, method=:max)
            
            # Compute width
            width, _, _ = FailureOfInhibition2025.compute_half_max_width(sol, 1, nothing, current_params.lattice)
            
            # Compute decay rate
            decay_rate, _ = FailureOfInhibition2025.compute_decay_rate(sol, 1)
            
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
            @warn "GPU simulation failed with parameters: $x" exception=e
            return 1e6
        end
    end
    
    # Run optimization with bounds
    result = Optim.optimize(
        obj_function,
        lower_bounds,
        upper_bounds,
        x0,
        Optim.Fminbox(method),
        Optim.Options(iterations=maxiter, show_trace=false)
    )
    
    # Create best parameters
    best_x = Optim.minimizer(result)
    best_params = FailureOfInhibition2025._update_params(base_params, param_names, best_x)
    
    return result, best_params
end

end # module FailureOfInhibition2025CUDAExt
