include(joinpath(@__DIR__,"..","scripts","catalog_classifications.jl"))
@testset "Catalog classification proposals" begin
    M=CatalogClassifications; C=M.C; p=M.POLICIES[2]
    d(e,i;arm=1,branch=1,stability=1,near=false,omega=0.,a=-.1,b=-.2) = C.Descriptor(1,1,arm,branch,e,i,a,b,omega,stability,near)
    @test M.candidate(d(.01,.01;arm=2),p)==1
    @test M.candidate(d(.02,0.;arm=0,branch=2),p)==2
    @test M.candidate(d(.49,.3;branch=2),p)==3
    @test M.candidate(d(.49,.3;branch=1),p)==3
    @test M.candidate(d(.45,.02;branch=2),p)==4
    @test M.candidate(d(.45,.01;branch=2),p)==2
    @test M.candidate(d(.45,.3;branch=2),p)==3
    @test M.candidate(d(.49,0.;branch=1),p)==0
    @test M.candidate(d(.49,0.;branch=3),p)==0
    @test M.candidate(d(.49,0.;branch=0),p)==0
    @test M.candidate(d(.001,.49),p)==0
    @test M.candidate(d(.1,.1),p)==0
    @test M.candidate(d(.49,0.;branch=2,stability=2,a=.1),p)==2
    @test !M.eligible(d(.49,0.;branch=2,stability=2,a=.1))
    @test !M.eligible(d(.01,.01;near=true))
    @test M.activity_code(d(.01,.01),p)==1
    @test M.activity_code(d(.45,.30),p)==9
    @test M.activity_name(2)=="low E / intermediate I"
    @test M.annotation(d(.5,0.;branch=2)).agreement
    @test !M.annotation(d(.015,.005)).agreement
    @test_throws ArgumentError M.Policy("invalid",.1,.01,.05,.3)
    @test_throws ArgumentError M.Policy("invalid",NaN,.01,.4,.3)
    c=(;context_id=1,scope="fixture",case_id="fixture",B_E=0.,B_I=0.)
    s=M.summarize(c,[d(.49,0.;branch=2),d(.48,0.;branch=2),d(.4,.4;stability=2,a=.1)])
    @test s.sinks==2 && s.seizure==2 && s.roots==3
    @test s.E_min==.48 && s.E_max==.49 && s.I_min==s.I_max==0.
    empty=M.summarize(c,C.Descriptor[])
    @test empty.sinks==0 && ismissing(empty.E_min) && ismissing(empty.alpha_max)
    unstable=M.summarize(c,[d(.4,.4;stability=2,a=.1)])
    @test unstable.sinks==0 && unstable.roots==1 && ismissing(unstable.I_max)
    # Geometry is an independent qualifier, including nonfolded and unresolved E geometry.
    for arm in (0,1,2,3,4,255)
        @test M.candidate(d(.001,.001;arm),p)==1
        @test M.candidate(d(.49,0.;arm,branch=2),p)==2
        @test M.candidate(d(.49,.49;arm),p)==3
    end
    # Cross each declared boundary by one representable value.
    for policy in M.POLICIES
        @test M.candidate(d(policy.low_E,policy.low_I),policy)==1
        @test M.candidate(d(nextfloat(policy.low_E),policy.low_I;branch=1),policy)==0
        @test M.candidate(d(nextfloat(policy.low_E),policy.low_I;branch=2),policy)==2
        @test M.candidate(d(policy.high_E,policy.high_I),policy)==3
        @test M.candidate(d(prevfloat(policy.high_E),policy.high_I),policy)==0
        @test M.candidate(d(policy.high_E,prevfloat(policy.high_I);branch=2),policy)==4
        @test M.candidate(d(policy.high_E,policy.low_I;branch=2),policy)==2
    end
end
