using Test
include(joinpath(@__DIR__,"..","scripts","equilibrium_nomenclature.jl"))
@testset "Independent exploratory equilibrium memberships" begin
    N = EquilibriumNomenclature
    b = N.Boundaries(.02,.02,.02,.01,.4,.3)
    @test N.memberships(.01,.01,"ascending",b).rest
    @test !N.memberships(.01,.4,"ascending",b).rest
    @test !N.memberships(.01,.001,"descending",b).seizure
    @test N.memberships(.49,.001,"descending",b).seizure
    @test !N.memberships(.49,.001,"ascending",b).seizure
    @test !N.memberships(.49,.001,"peak",b).seizure
    @test N.memberships(.49,.4,"ascending",b).herald
    @test N.memberships(.49,.4,"descending",b).herald
    @test N.memberships(.49,.15,"descending",b).partial_failure
    @test !N.memberships(.49,.15,"ascending",b).partial_failure
    @test N.memberships(.02,.02,"ascending",b).rest
    @test !N.memberships(.02,.001,"descending",b).seizure
    @test N.memberships(.49,.01,"descending",b).seizure
    @test !N.memberships(.49,.01,"descending",b).partial_failure
    @test N.memberships(.4,.3,"descending",b).herald
    @test !N.memberships(.4,.3,"descending",b).partial_failure
    # A deliberate overlap is retained, never resolved by precedence.
    overlap=N.memberships(.015,.001,"descending",N.Boundaries(.02,.02,.01,.01,.4,.3))
    @test overlap.rest && overlap.seizure
    @test sum(values(N.memberships(.2,.2,"ascending",b)))==0
    # No context, stability, E arm, or unique-root requirement enters membership.
    population=[(.49,.001,"descending"),(.48,.002,"descending")]
    original=[N.memberships(x...,b) for x in population]
    @test all(m->m.seizure,original)
    extended=[N.memberships(x...,b) for x in vcat(population,[(.01,.01,"ascending")])]
    @test original==extended[1:2]
    @test N.memberships(.015,.01,"ascending",b).rest # possible middle-arm coordinates
    @test_throws ArgumentError N.Boundaries(NaN,0,0,0,0,0)
    @test_throws ArgumentError N.Boundaries(-.1,0,0,0,0,0)
    @test_throws ArgumentError N.memberships(.6,0.,"ascending",b)
    @test_throws ArgumentError N.memberships(.1,.1,"unknown",b)
    xs=[0.,.1,.1,.5]
    @test [N.observed_quantile(xs,q) for q in (0.,.25,.5,1.)]==[0.,0.,.1,.5]
    @test_throws ArgumentError N.observed_quantile(Float64[],.5)
    E=[0.,.1,.1,.3,.5];I=[.5,.1,0.,.2,.1]
    boundaries=[0.,.1,.1,.2,.5]
    counts=N.lower_counts(E,I,boundaries,boundaries)
    oracle=[count(k->E[k]<=e && I[k]<=i,eachindex(E)) for e in boundaries,i in boundaries]
    @test counts==oracle
    high=reverse(N.lower_counts(-E,-I,reverse(-boundaries),reverse(-boundaries));dims=(1,2))
    @test high==[count(k->E[k]>=e && I[k]>=i,eachindex(E)) for e in boundaries,i in boundaries]
    @test_throws DimensionMismatch N.lower_counts(E,I[1:2],boundaries,boundaries)
    @test_throws ArgumentError N.lower_counts(E,I,reverse(boundaries),boundaries)
end
