module FailureOfInhibition2025CUDAExt

using FailureOfInhibition2025
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using Statistics
using LinearAlgebra

# Override the CUDA detection function
function FailureOfInhibition2025._cuda_is_functional()
    return CUDA.functional()
end

"""
    FailureOfInhibition2025.GaussianConnectivity(param::GaussianConnectivityParameter, lattice, ::Type{CuArray})

GPU-specific constructor for GaussianConnectivity that uses CUDA FFT plans.

This constructor should be called once before running GPU simulations to create
GaussianConnectivity objects with GPU-compatible FFT plans and buffers.

# Arguments
- `param`: GaussianConnectivityParameter with amplitude and spread
- `lattice`: Spatial lattice for the connectivity kernel
- `::Type{CuArray}`: Type marker to dispatch to GPU version

# Returns
- `GaussianConnectivity` object with CUDA FFT plans and GPU buffers

# Example
```julia
using CUDA

# Create GPU-compatible connectivity
param = GaussianConnectivityParameter(1.0, (2.0,))
lattice = CompactLattice(extent=(10.0,), n_points=(101,))
gc_gpu = GaussianConnectivity(param, lattice, CuArray)

# Use in ConnectivityMatrix
connectivity = ConnectivityMatrix{1}([gc_gpu])
```
"""
function FailureOfInhibition2025.GaussianConnectivity(
    param::FailureOfInhibition2025.GaussianConnectivityParameter, 
    lattice,
    ::Type{CuArray}
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
    solve_model(initial_condition, tspan, params::WilsonCowanParameters{N}; solver=Tsit5(), use_gpu=nothing, kwargs...) where N

GPU-accelerated version of solve_model (CUDA extension override).

When use_gpu is true or nothing (and CUDA is functional), this method automatically 
transfers data to GPU, runs the simulation, and transfers results back to CPU.
Any GaussianConnectivityParameter objects in the connectivity matrix are automatically
converted to GPU-compatible GaussianConnectivity objects with CUDA FFT plans before
simulation begins, ensuring FFT plans are created once (not on every timestep).
"""
function FailureOfInhibition2025.solve_model(
    initial_condition, 
    tspan, 
    params::FailureOfInhibition2025.WilsonCowanParameters{N}; 
    solver=Tsit5(), 
    use_gpu=nothing, 
    kwargs...
) where N
    # Determine whether to use GPU
    should_use_gpu = if use_gpu === nothing
        CUDA.functional()
    elseif use_gpu === true
        if !CUDA.functional()
            error("GPU acceleration requested but CUDA is not functional")
        end
        true
    else
        false
    end
    
    # If not using GPU, call the base implementation
    if !should_use_gpu
        # Call the base CPU implementation by constructing the problem directly
        prob = ODEProblem(FailureOfInhibition2025.wcm1973!, initial_condition, tspan, params)
        return solve(prob, solver; kwargs...)
    end
    
    # GPU implementation
    # Transfer initial condition to GPU
    u0_gpu = CuArray(initial_condition)
    
    # Convert connectivity to GPU-compatible version (if needed)
    gpu_params = convert_connectivity_to_gpu(params)
    
    # Create ODE problem on GPU
    prob = ODEProblem(FailureOfInhibition2025.wcm1973!, u0_gpu, tspan, gpu_params)
    
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
    convert_connectivity_to_gpu(params::WilsonCowanParameters)

Convert any GaussianConnectivityParameter objects in the connectivity matrix
to GPU-compatible GaussianConnectivity objects with CUDA FFT plans.

This ensures FFT plans are created once before simulation, not on every timestep.
"""
function convert_connectivity_to_gpu(params::FailureOfInhibition2025.WilsonCowanParameters{T,P}) where {T,P}
    connectivity = params.connectivity
    
    # If connectivity is a ConnectivityMatrix, convert GaussianConnectivityParameter to GaussianConnectivity
    if connectivity isa FailureOfInhibition2025.ConnectivityMatrix
        gpu_matrix = map(connectivity.matrix) do conn
            if conn isa FailureOfInhibition2025.GaussianConnectivityParameter
                # Convert to GPU-compatible GaussianConnectivity
                return FailureOfInhibition2025.GaussianConnectivity(conn, params.lattice, CuArray)
            else
                # Keep other connectivity types as-is
                return conn
            end
        end
        
        gpu_connectivity = FailureOfInhibition2025.ConnectivityMatrix{P}(gpu_matrix)
        
        # Create new params with GPU connectivity
        return FailureOfInhibition2025.WilsonCowanParameters{T,P}(
            params.α, params.β, params.τ,
            gpu_connectivity,
            params.nonlinearity, params.stimulus, params.lattice, params.pop_names
        )
    else
        # If connectivity is not a ConnectivityMatrix, return params unchanged
        return params
    end
end



end # module FailureOfInhibition2025CUDAExt
