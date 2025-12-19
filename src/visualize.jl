"""
Visualization utilities for Wilson-Cowan model simulations.

Provides tools to visualize simulations on various lattices:
- Time series plots for point models
- Line plots and animations for 1D lattices
- Heatmaps and animated heatmaps for 2D lattices
- Spatiotemporal plots (space vs time)
"""

using Plots
using Statistics

"""
    plot_time_series(sol, params::WilsonCowanParameters{T,N}; 
                     pop_indices=nothing, title="Population Activity Over Time",
                     kwargs...) where {T,N}

Plot time series of population activities from a simulation solution.

Works for all lattice types (PointLattice, 1D, 2D). For spatial models,
plots the mean activity across space for each population.

# Arguments
- `sol`: ODE solution from `solve_model()`
- `params`: WilsonCowanParameters containing model configuration
- `pop_indices`: Vector of population indices to plot (default: all populations)
- `title`: Plot title
- `kwargs...`: Additional arguments passed to `plot()`

# Returns
- Plots.Plot object

# Examples
```julia
# Point model
sol = solve_model(A₀, tspan, params)
p = plot_time_series(sol, params)

# Spatial model (plots mean activity)
sol = solve_model(A₀, tspan, params_spatial)
p = plot_time_series(sol, params_spatial, pop_indices=[1, 2])
```
"""
function plot_time_series(sol, params::WilsonCowanParameters{T,N}; 
                         pop_indices=nothing, 
                         title="Population Activity Over Time",
                         kwargs...) where {T,N}
    # Determine which populations to plot
    pops_to_plot = pop_indices === nothing ? collect(1:N) : pop_indices
    
    # Create plot
    p = plot(; title=title, xlabel="Time", ylabel="Activity", 
             legend=:topright, kwargs...)
    
    # Extract time points
    times = sol.t
    
    # For each population
    for pop_idx in pops_to_plot
        # Extract activity over time
        if ndims(sol.u[1]) == 1
            # Point model: simple 1D array
            activity = [u[pop_idx] for u in sol.u]
        elseif size(sol.u[1], 1) == 1
            # Point model with connectivity: (1, N_pop)
            activity = [u[1, pop_idx] for u in sol.u]
        else
            # Spatial model: compute mean across space
            activity = [mean(u[:, pop_idx]) for u in sol.u]
        end
        
        # Plot
        pop_name = params.pop_names[pop_idx]
        plot!(p, times, activity, label=pop_name, linewidth=2)
    end
    
    return p
end

"""
    plot_spatial_snapshot(sol, params::WilsonCowanParameters{T,N}, time_idx::Int;
                         pop_idx=1, title="Spatial Activity",
                         kwargs...) where {T,N}

Plot a snapshot of spatial activity at a specific time point.

Works for 1D and 2D lattices. For 1D, creates a line plot. For 2D, creates a heatmap.

# Arguments
- `sol`: ODE solution from `solve_model()`
- `params`: WilsonCowanParameters containing model configuration
- `time_idx`: Index into sol.t for the time point to plot
- `pop_idx`: Population index to plot (default: 1)
- `title`: Plot title
- `kwargs...`: Additional arguments passed to `plot()` or `heatmap()`

# Returns
- Plots.Plot object

# Examples
```julia
# 1D lattice
sol = solve_model(A₀, tspan, params_1d, saveat=0.1)
p = plot_spatial_snapshot(sol, params_1d, 50, pop_idx=1)

# 2D lattice
sol = solve_model(A₀, tspan, params_2d, saveat=0.1)
p = plot_spatial_snapshot(sol, params_2d, 50, pop_idx=1)
```
"""
function plot_spatial_snapshot(sol, params::WilsonCowanParameters{T,N}, time_idx::Int;
                              pop_idx=1, 
                              title="Spatial Activity",
                              kwargs...) where {T,N}
    # Get lattice
    lattice = params.lattice
    
    # Get state at specified time
    state = sol.u[time_idx]
    activity = state[:, pop_idx]
    
    # Get dimensionality
    n_dims = ndims(lattice)
    
    if n_dims == 0
        error("Cannot plot spatial snapshot for PointLattice (0D). Use plot_time_series instead.")
    elseif n_dims == 1
        # 1D lattice: line plot
        coords = coordinates(lattice)
        x_coords = [c[1] for c in coords]
        
        p = plot(x_coords, activity; 
                title=title,
                xlabel="Position", 
                ylabel="Activity",
                label=params.pop_names[pop_idx],
                linewidth=2,
                legend=:topright,
                kwargs...)
    elseif n_dims == 2
        # 2D lattice: heatmap
        activity_matrix = reshape(activity, size(lattice))
        coords = coordinates(lattice)
        
        # Extract x and y coordinates
        x_coords = [coords[1, j][1] for j in 1:size(coords, 2)]
        y_coords = [coords[i, 1][2] for i in 1:size(coords, 1)]
        
        p = heatmap(x_coords, y_coords, activity_matrix;
                   title=title,
                   xlabel="X", 
                   ylabel="Y",
                   colorbar_title="Activity",
                   c=:viridis,
                   kwargs...)
    else
        error("Unsupported lattice dimensionality: $n_dims")
    end
    
    # Add time information to title
    time_val = sol.t[time_idx]
    plot!(p, title="$title (t=$(round(time_val, digits=2)))")
    
    return p
end

"""
    plot_spatiotemporal(sol, params::WilsonCowanParameters{T,N};
                       pop_idx=1, title="Spatiotemporal Activity",
                       kwargs...) where {T,N}

Create a spatiotemporal plot (space vs time heatmap) for 1D lattices.

Shows how activity evolves over both space and time as a 2D heatmap where
x-axis is spatial position and y-axis is time.

# Arguments
- `sol`: ODE solution from `solve_model()`
- `params`: WilsonCowanParameters containing model configuration
- `pop_idx`: Population index to plot (default: 1)
- `title`: Plot title
- `kwargs...`: Additional arguments passed to `heatmap()`

# Returns
- Plots.Plot object

# Examples
```julia
# 1D lattice
lattice = CompactLattice(extent=(20.0,), n_points=(101,))
params = WilsonCowanParameters{1}(...)
sol = solve_model(A₀, tspan, params, saveat=0.1)
p = plot_spatiotemporal(sol, params)
```
"""
function plot_spatiotemporal(sol, params::WilsonCowanParameters{T,N};
                            pop_idx=1,
                            title="Spatiotemporal Activity",
                            kwargs...) where {T,N}
    # Get lattice
    lattice = params.lattice
    n_dims = ndims(lattice)
    
    if n_dims != 1
        error("Spatiotemporal plots are only supported for 1D lattices. Got $(n_dims)D.")
    end
    
    # Extract spatial and temporal dimensions
    n_spatial = length(sol.u[1][:, pop_idx])
    n_time = length(sol.t)
    
    # Create spatiotemporal activity matrix (time x space)
    activity_matrix = zeros(n_time, n_spatial)
    for (t_idx, state) in enumerate(sol.u)
        activity_matrix[t_idx, :] = state[:, pop_idx]
    end
    
    # Get spatial coordinates
    coords = coordinates(lattice)
    x_coords = [c[1] for c in coords]
    
    # Create heatmap
    p = heatmap(x_coords, sol.t, activity_matrix;
               title=title,
               xlabel="Space", 
               ylabel="Time",
               colorbar_title="Activity",
               c=:viridis,
               kwargs...)
    
    return p
end

"""
    animate_1d(sol, params::WilsonCowanParameters{T,N};
               pop_idx=1, filename="animation.gif",
               fps=10, ylims=nothing,
               kwargs...) where {T,N}

Create an animation of activity evolution for 1D lattices.

# Arguments
- `sol`: ODE solution from `solve_model()`
- `params`: WilsonCowanParameters containing model configuration
- `pop_idx`: Population index to animate (default: 1)
- `filename`: Output filename for the animation (default: "animation.gif")
- `fps`: Frames per second (default: 10)
- `ylims`: Y-axis limits as tuple (ymin, ymax), or nothing for auto
- `kwargs...`: Additional arguments passed to `plot()`

# Returns
- Animation object

# Examples
```julia
# 1D lattice
lattice = CompactLattice(extent=(20.0,), n_points=(101,))
params = WilsonCowanParameters{1}(...)
sol = solve_model(A₀, tspan, params, saveat=0.1)
anim = animate_1d(sol, params, filename="wave.gif", fps=20)
```
"""
function animate_1d(sol, params::WilsonCowanParameters{T,N};
                   pop_idx=1,
                   filename="animation.gif",
                   fps=10,
                   ylims=nothing,
                   kwargs...) where {T,N}
    # Get lattice
    lattice = params.lattice
    n_dims = ndims(lattice)
    
    if n_dims != 1
        error("1D animation is only supported for 1D lattices. Got $(n_dims)D.")
    end
    
    # Get coordinates
    coords = coordinates(lattice)
    x_coords = [c[1] for c in coords]
    
    # Determine y-axis limits if not provided
    if ylims === nothing
        all_activities = [state[:, pop_idx] for state in sol.u]
        global_min = minimum(minimum.(all_activities))
        global_max = maximum(maximum.(all_activities))
        margin = (global_max - global_min) * 0.1
        ylims = (global_min - margin, global_max + margin)
    end
    
    # Create animation
    anim = @animate for (t_idx, state) in enumerate(sol.u)
        activity = state[:, pop_idx]
        time_val = sol.t[t_idx]
        
        plot(x_coords, activity;
            title="$(params.pop_names[pop_idx]) Activity (t=$(round(time_val, digits=2)))",
            xlabel="Position",
            ylabel="Activity",
            ylims=ylims,
            legend=false,
            linewidth=2,
            kwargs...)
    end
    
    # Save animation
    gif(anim, filename, fps=fps)
    
    return anim
end

"""
    animate_2d(sol, params::WilsonCowanParameters{T,N};
               pop_idx=1, filename="animation.gif",
               fps=10, clims=nothing,
               kwargs...) where {T,N}

Create an animated heatmap of activity evolution for 2D lattices.

# Arguments
- `sol`: ODE solution from `solve_model()`
- `params`: WilsonCowanParameters containing model configuration
- `pop_idx`: Population index to animate (default: 1)
- `filename`: Output filename for the animation (default: "animation.gif")
- `fps`: Frames per second (default: 10)
- `clims`: Color limits as tuple (cmin, cmax), or nothing for auto
- `kwargs...`: Additional arguments passed to `heatmap()`

# Returns
- Animation object

# Examples
```julia
# 2D lattice
lattice = CompactLattice(extent=(10.0, 10.0), n_points=(51, 51))
params = WilsonCowanParameters{1}(...)
sol = solve_model(A₀, tspan, params, saveat=0.1)
anim = animate_2d(sol, params, filename="pattern.gif", fps=15)
```
"""
function animate_2d(sol, params::WilsonCowanParameters{T,N};
                   pop_idx=1,
                   filename="animation.gif",
                   fps=10,
                   clims=nothing,
                   kwargs...) where {T,N}
    # Get lattice
    lattice = params.lattice
    n_dims = ndims(lattice)
    
    if n_dims != 2
        error("2D animation is only supported for 2D lattices. Got $(n_dims)D.")
    end
    
    # Get coordinates
    coords = coordinates(lattice)
    x_coords = [coords[1, j][1] for j in 1:size(coords, 2)]
    y_coords = [coords[i, 1][2] for i in 1:size(coords, 1)]
    
    # Determine color limits if not provided
    if clims === nothing
        all_activities = [state[:, pop_idx] for state in sol.u]
        global_min = minimum(minimum.(all_activities))
        global_max = maximum(maximum.(all_activities))
        clims = (global_min, global_max)
    end
    
    # Create animation
    anim = @animate for (t_idx, state) in enumerate(sol.u)
        activity = state[:, pop_idx]
        activity_matrix = reshape(activity, size(lattice))
        time_val = sol.t[t_idx]
        
        heatmap(x_coords, y_coords, activity_matrix;
               title="$(params.pop_names[pop_idx]) Activity (t=$(round(time_val, digits=2)))",
               xlabel="X",
               ylabel="Y",
               colorbar_title="Activity",
               c=:viridis,
               clims=clims,
               kwargs...)
    end
    
    # Save animation
    gif(anim, filename, fps=fps)
    
    return anim
end

"""
    plot_multi_population_snapshot(sol, params::WilsonCowanParameters{T,N}, time_idx::Int;
                                  layout=nothing, title="Multi-Population Activity",
                                  kwargs...) where {T,N}

Plot spatial snapshots of all populations at a specific time point.

Creates a subplot for each population. Works for 1D and 2D lattices.

# Arguments
- `sol`: ODE solution from `solve_model()`
- `params`: WilsonCowanParameters containing model configuration
- `time_idx`: Index into sol.t for the time point to plot
- `layout`: Plot layout (e.g., (2,2) for 2x2 grid), or nothing for automatic
- `title`: Overall title
- `kwargs...`: Additional arguments passed to individual plots

# Returns
- Plots.Plot object with subplots

# Examples
```julia
# 1D lattice with 2 populations
sol = solve_model(A₀, tspan, params, saveat=0.1)
p = plot_multi_population_snapshot(sol, params, 50)
```
"""
function plot_multi_population_snapshot(sol, params::WilsonCowanParameters{T,N}, time_idx::Int;
                                       layout=nothing,
                                       title="Multi-Population Activity",
                                       kwargs...) where {T,N}
    # Determine layout if not provided
    if layout === nothing
        # Try to make roughly square
        ncols = ceil(Int, sqrt(N))
        nrows = ceil(Int, N / ncols)
        layout = (nrows, ncols)
    end
    
    # Create subplots for each population
    plots = []
    for pop_idx in 1:N
        pop_title = "$(params.pop_names[pop_idx])"
        p = plot_spatial_snapshot(sol, params, time_idx; 
                                 pop_idx=pop_idx, 
                                 title=pop_title,
                                 kwargs...)
        push!(plots, p)
    end
    
    # Combine into single figure
    time_val = sol.t[time_idx]
    combined = plot(plots...; layout=layout, 
                   plot_title="$title (t=$(round(time_val, digits=2)))",
                   size=(400*layout[2], 300*layout[1]))
    
    return combined
end

"""
    plot_phase_portrait(sol, params::WilsonCowanParameters{2};
                       title="Phase Portrait",
                       kwargs...)

Plot phase portrait for 2-population models.

Shows trajectory in (E, I) phase space. Works for point models and spatial models
(in which case it plots the trajectory of spatially-averaged activities).

# Arguments
- `sol`: ODE solution from `solve_model()`
- `params`: WilsonCowanParameters with 2 populations
- `title`: Plot title
- `kwargs...`: Additional arguments passed to `plot()`

# Returns
- Plots.Plot object

# Examples
```julia
# Point model
params = create_point_model_wcm1973(:oscillatory)
sol = solve_model(A₀, tspan, params)
p = plot_phase_portrait(sol, params)
```
"""
function plot_phase_portrait(sol, params::WilsonCowanParameters{2};
                            title="Phase Portrait",
                            kwargs...)
    # Extract activities for both populations
    if ndims(sol.u[1]) == 1
        # Point model: simple 1D array
        E_activity = [u[1] for u in sol.u]
        I_activity = [u[2] for u in sol.u]
    elseif size(sol.u[1], 1) == 1
        # Point model with connectivity: (1, 2)
        E_activity = [u[1, 1] for u in sol.u]
        I_activity = [u[1, 2] for u in sol.u]
    else
        # Spatial model: use mean across space
        E_activity = [mean(u[:, 1]) for u in sol.u]
        I_activity = [mean(u[:, 2]) for u in sol.u]
    end
    
    # Create phase portrait
    p = plot(E_activity, I_activity;
            title=title,
            xlabel=params.pop_names[1],
            ylabel=params.pop_names[2],
            label="Trajectory",
            linewidth=2,
            legend=:topright,
            kwargs...)
    
    # Mark start and end points
    scatter!(p, [E_activity[1]], [I_activity[1]], 
            label="Start", markersize=8, markercolor=:green)
    scatter!(p, [E_activity[end]], [I_activity[end]], 
            label="End", markersize=8, markercolor=:red)
    
    return p
end
