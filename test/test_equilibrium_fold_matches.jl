using Test
include(joinpath(@__DIR__,"..","scripts","check_equilibrium_fold_matches.jl"))
@testset "Fold comparison retains both neighboring correction seeds" begin
    point(p,e;direction=-1)=(;variant="half_step",direction,parameter=p,E=e,I=.2)
    # A fold crosses the same parameter on two different segments.
    points=[point(0.,.1),point(1.,.11),point(0.,.12)]
    seeds=EquilibriumFoldMatches.candidates(points,.5,[.11,.2])
    @test [.1,.2] in seeds
    @test [.11,.2] in seeds
    @test [.12,.2] in seeds
    @test any(s->isapprox(s[1],.105),seeds)
    @test any(s->isapprox(s[1],.115),seeds)
    @test isempty(EquilibriumFoldMatches.candidates(points,2.,[.11,.2]))
    @test isempty(EquilibriumFoldMatches.candidates(points,.5,[.4,.4]))
end
