@testset "Shared recurrence basin map" begin
    model=PointModelParameters(
        excitatory=PopulationParameters(timescale=1.,response=LogisticResponse(slope=1.,threshold=0.)),
        inhibitory=PopulationParameters(timescale=1.,response=LogisticResponse(slope=1.,threshold=0.)),
        coupling=PointCoupling(0.,0.,0.,0.))
    search=find_equilibria(model)
    po=PulseExperimentOptions(followup_times=[50.],
        diagnostic_options=DiagnosticOptions(window_duration=3.,min_samples=7))
    opts=BasinGridOptions(grid_points=15,pulse_options=po)
    result=basin_grid(model,search;options=opts)
    @test first(result.E)==first(result.I)==0.
    @test last(result.E)==last(result.I)==.5
    @test size(result.labels)==(15,15)
    @test all(==(1),result.labels)
    @test all(==(:compatible),result.statuses)
    @test result.statistics.cached_starts>0
    @test result.statistics.basin_hits>0
    @test result.statistics.total_steps<15^2*100
    repeated=basin_grid(model,search;options=opts)
    @test result.labels==repeated.labels
    @test result.statistics.iterations==repeated.statistics.iterations
    # A displayed region need not contain the attractor. Its orbits can leave
    # that region and still be resolved in the physical-domain support grid.
    outside=basin_grid(model,search;options=BasinGridOptions(grid_points=7,
        bounds=((0.,.1),(0.,.1)),pulse_options=po))
    @test all(==(1),outside.labels)
    @test only(search.equilibria).state[1]>.1
    capped=basin_grid(model,search;options=BasinGridOptions(grid_points=3,
        maximum_iterations=1,pulse_options=po))
    @test all(==(0),capped.labels)
    @test all(!=(:compatible),capped.statuses)
    for kwargs in ((;grid_points=true),(;grid_points=1),(;sample_dt=0),
        (;bounds=((0.,0.),(0.,.5))),(;consecutive_basin_steps=0))
        @test_throws ArgumentError BasinGridOptions(;kwargs...)
    end
end

@testset "Basin boundaries and interrupted attractor discovery" begin
    model=PointModelParameters(
        excitatory=PopulationParameters(timescale=1.,response=LogisticResponse(slope=1.,threshold=3+log(2))),
        inhibitory=PopulationParameters(timescale=1.,response=LogisticResponse(slope=1.,threshold=0.)),
        coupling=PointCoupling(12.,0.,0.,0.))
    search=find_equilibria(model;seeds=[[e,1/3] for e in (0.,.25,.5)])
    @test length(search.equilibria)==3
    po=PulseExperimentOptions(followup_times=[100.],
        diagnostic_options=DiagnosticOptions(window_duration=3.,min_samples=7))
    grid=basin_grid(model,search;options=BasinGridOptions(grid_points=13,pulse_options=po))
    @test all(==(0),grid.labels[7,:])
    @test all(==(1),grid.labels[1,:])
    @test all(==(3),grid.labels[end,:])
    @test any(==(:boundary_unresolved),grid.statuses)
    for index in CartesianIndices(grid.labels)
        grid.labels[index]==0 && continue
        independent=basin_destination(model,search,[grid.E[index[1]],grid.I[index[2]]];options=po)
        @test independent.status==:compatible
        @test independent.destination==grid.labels[index]
    end
    # Stop after recurrence discovery begins but before the attractor-location
    # phase completes. Provisional cells/sets must not survive for later starts.
    incomplete=basin_grid(model,search;options=BasinGridOptions(grid_points=7,
        maximum_iterations=105,pulse_options=po))
    @test all(==(0),incomplete.labels)
    @test isempty(incomplete.statistics.recurrence_to_equilibrium)
end
