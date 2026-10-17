include(joinpath(@__DIR__,"..","scripts","catalog_structural_distributions.jl"))
@testset "Structural occupancy distributions" begin
    M=CatalogStructuralDistributions
    s=M.distribution([.5,0.,.01,0.])
    @test s.n==4 && s.zeros==2 && s.ceiling==1
    @test s.quantiles==[0.,0.,0.,0.,.01,.5,.5]
    @test s.counts[1]==2 && s.counts[2]==1 && s.counts[50]==1 && sum(s.counts)==4
    @test M.distribution([.5]).quantiles==fill(.5,7)
    @test M.distribution([0.]).counts[1]==1
    @test M.distribution([prevfloat(.01),.01,nextfloat(.01)]).counts[1:2]==[1,2]
    empty=M.distribution(Float64[])
    @test empty.n==0 && all(ismissing,empty.quantiles) && sum(empty.counts)==0
    @test_throws ArgumentError M.distribution([NaN])
    @test_throws ArgumentError M.distribution([-.001])
    @test_throws ArgumentError M.ecdf_steps([.50001])
    x,y=M.ecdf_steps([.5,0.,.01,0.])
    @test x==[0.,0.,0.,.01,.01,.5,.5,.5]
    @test y==[0.,0.,.5,.5,.75,.75,1.,1.]
    @test M.ecdf_steps(Float64[])==(Float64[],Float64[])
    @test M.ecdf_steps([.2,.1])==M.ecdf_steps([.1,.2])
    d(status,near)=M.C.Descriptor(1,1,2,1,.1,.2,-.1,-.2,0.,status,near)
    @test M.eligible(d(1,false))
    @test !M.eligible(d(1,true)) && !M.eligible(d(0,false)) && !M.eligible(d(2,false))
    h=M.joint_histogram([0.,.5,.005],[.5,0.,.005])
    @test sum(h)==3 && h[1,100]==1 && h[100,1]==1 && h[2,2]==1
    @test M.joint_histogram([.1,.2],[.3,.4])==transpose(M.joint_histogram([.3,.4],[.1,.2]))
    @test sum(M.joint_histogram(Float64[],Float64[]))==0
    @test_throws DimensionMismatch M.joint_histogram([0.],Float64[])
    @test_throws ArgumentError M.joint_histogram([0.],[Inf])

end
