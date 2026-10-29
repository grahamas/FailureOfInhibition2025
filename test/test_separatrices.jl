function separatrix_test_model(::Type{T}=Float64; coupled=false, recurrent=12) where {T}
    return PointModelParameters(
        excitatory=PopulationParameters(timescale=one(T),
            response=LogisticResponse(slope=one(T),
                threshold=T(3 + log(2) - (coupled ? 1 / 3 : 0)))),
        inhibitory=PopulationParameters(timescale=one(T),
            response=LogisticResponse(slope=one(T), threshold=T(coupled ? 0.25 : 0))),
        coupling=PointCoupling(T(recurrent), T(coupled), T(coupled), zero(T)))
end

@testset "Separatrix numerical policy" begin
    @test SeparatrixOptions().geometry_atol == 0.5 / 512
    @test SeparatrixOptions(display_span=1, display_pixels=100).geometry_atol == 0.01
    for kwargs in ((; seed_offset=0), (; seed_offset=1), (; maximum_time=Inf),
        (; equilibrium_radius=-1), (; abstol=NaN), (; reltol=true),
        (; tightening_factor=1), (; geometry_atol=0), (; display_span=0),
        (; display_pixels=true), (; maxiters=0), (; max_points=1))
        @test_throws ArgumentError SeparatrixOptions(; kwargs...)
    end
end

@testset "Two tangent saddle branches and physical boundary events" begin
    for T in (Float64, Float32)
        model = separatrix_test_model(T)
        search = find_equilibria(model; seeds=[T[0.25, 1 / 3]],
            options=T === Float32 ? EquilibriumOptions(solver_abstol=1f-6,
                solver_reltol=1f-5, residual_atol=1f-6) : EquilibriumOptions())
        @test only(search.equilibria).stability.classification == Saddle
        branches = trace_separatrices(model, search;
            options=SeparatrixOptions(maximum_time=100))
        @test length(branches) == 2
        @test [branch.sign for branch in branches] == [-1, 1]
        for branch in branches
            @test branch.root_index == 1
            @test branch.converged
            @test branch.status == :converged
            @test branch.termination_reason == :physical_boundary
            @test length(branch.attempts) == 2
            @test branch.attempts[2].seed_offset == branch.attempts[1].seed_offset / 2
            @test branch.attempts[2].abstol < branch.attempts[1].abstol
            @test branch.attempts[2].reltol < branch.attempts[1].reltol
            @test eltype(first(branch.points)) === T
            @test branch.geometric_disagreement <= branch.geometry_atol
            @test all(isapprox.(branch.E, T(0.25); atol=T(2e-6)))
            @test isapprox(last(branch.I), branch.sign < 0 ? 0 : 1; atol=32eps(T))
            @test all(point -> all(value -> -32eps(T) <= value <= 1 + 32eps(T), point), branch.points)
            displacement = branch.points[2] - branch.root_state
            @test isapprox(displacement / sqrt(sum(abs2, displacement)),
                branch.sign * branch.stable_eigenvector; atol=T(0.01))
            @test branch.stable_eigenvalue < 0
            # Continuous events reach the edge in finite analysis time rather
            # than repeatedly rejecting steps just outside the domain.
            @test 0 < branch.attempts[2].analysis_time < 30
        end
    end
end

@testset "Offset and tolerance convergence on curved stable manifolds" begin
    model = separatrix_test_model(; coupled=true)
    search = find_equilibria(model; seeds=[[0.25, 1 / 3]])
    @test only(search.equilibria).stability.classification == Saddle
    branches = trace_separatrices(model, search;
        options=SeparatrixOptions(seed_offset=1e-4, maximum_time=100))
    @test all(branch -> branch.converged, branches)
    @test all(branch -> branch.attempts[2].abstol < branch.attempts[1].abstol, branches)
    @test all(branch -> branch.geometric_disagreement < branch.geometry_atol, branches)
    @test any(branch -> maximum(branch.E) - minimum(branch.E) > 0.01, branches)
    # Tightening the comparison criterion exposes finite-offset disagreement;
    # diagnostic curves remain available, but no unconverged line is displayed.
    strict = trace_separatrices(model, search;
        options=SeparatrixOptions(seed_offset=0.2, maximum_time=100,
            geometry_atol=1e-5, max_points=200000))
    @test all(branch -> branch.status == :geometric_disagreement, strict)
    @test all(branch -> branch.converged || isempty(branch.points), strict)
end

@testset "Absent saddles, context validation, and explicit incomplete branches" begin
    model = separatrix_test_model(; recurrent=0)
    search = find_equilibria(model)
    @test isempty(trace_separatrices(model, search))
    @test_throws ArgumentError trace_separatrices(model, search; options=nothing)
    saddle = separatrix_test_model()
    saddle_search = find_equilibria(saddle; seeds=[[0.25, 1 / 3]])
    @test_throws ArgumentError trace_separatrices(model, saddle_search)
    multiple = find_equilibria(saddle)
    @test length(multiple.equilibria) == 3
    @test all(branch -> branch.converged, trace_separatrices(saddle, multiple;
        options=SeparatrixOptions(maximum_time=100)))
    short = trace_separatrices(saddle, saddle_search;
        options=SeparatrixOptions(maximum_time=0.01))
    @test all(branch -> !branch.converged && branch.status == :incomplete &&
        branch.termination_reason == :time_budget && isempty(branch.points) &&
        isempty(branch.E) && isempty(branch.I), short)
    outside = trace_separatrices(saddle, saddle_search;
        options=SeparatrixOptions(seed_offset=0.9, maximum_time=100))
    @test all(branch -> !branch.converged && isempty(branch.points), outside)
    @test all(branch -> branch.attempts[1].termination_reason == :seed_outside_domain, outside)
    exhausted = trace_separatrices(saddle, saddle_search;
        options=SeparatrixOptions(maximum_time=100, max_points=2))
    @test all(branch -> !branch.converged && branch.termination_reason == :point_budget,
        exhausted)
end

@testset "Stable manifold reaches another discovered equilibrium" begin
    # Independent bistability in E and I gives a repelling equilibrium between
    # two saddles on the horizontal I-unstable equilibrium line.
    model = PointModelParameters(
        excitatory=PopulationParameters(timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=3 + log(2))),
        inhibitory=PopulationParameters(timescale=1.0,
            response=FailureOfInhibitionResponse(slope=10.0,
                onset_threshold=0.0, failure_threshold=2.0)),
        coupling=PointCoupling(12.0, 0.0, 0.0, 10.0),
        drive=PiecewiseConstantDrive(baseline=(0.0, 4.0), pulses=(),
            interpretation=AfferentExcitation))
    search = find_equilibria(model; seeds=[[e, i]
        for e in (0.03, 0.25, 0.45) for i in (1e-8, 0.2, 0.4)])
    repeller = findfirst(root -> root.stability.classification == Repelling,
        search.equilibria)
    @test repeller !== nothing
    branches = trace_separatrices(model, search;
        options=SeparatrixOptions(maximum_time=100))
    connections = filter(branch -> branch.termination_reason == :equilibrium_neighborhood,
        branches)
    @test !isempty(connections)
    for branch in connections
        @test branch.converged
        @test branch.terminal_equilibrium == repeller
        @test isapprox(sqrt(sum(abs2, last(branch.points) -
            search.equilibria[repeller].state)), 1e-6; atol=1e-12)
    end
end
