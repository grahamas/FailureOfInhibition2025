using Test
include(joinpath(@__DIR__,"..","scripts","equilibrium_catalog.jl"))
@testset "Compact equilibrium descriptor contract" begin
    C=EquilibriumCatalog
    root(;E=.01,I=.005,arm="middle",side="ascending",s="Attracting",near=false,a=-.02,b=-.1,w=0.)=
        (;root=1,E,I,arm,I_side=side,stability=s,near_singular=near,
            lambda1_real=b,lambda1_imag=w==0 ? 0. : -w,lambda2_real=a,lambda2_imag=w)
    cases=[root(),root(a=.2,b=-.1,s="Saddle"),root(a=.2,b=.1,s="Repelling"),
        root(a=-.1,b=-.1,w=.4),root(a=0.,b=0.,w=.4,s="StabilityUnresolved",near=true),
        root(arm="no_turning_points",side="descending"),root(arm="turning_point",side="peak"),
        root(arm="unresolved",side="unresolved",s="StabilityUnresolved"),root(E=-0.,I=0.)]
    for r in cases
        d=C.encode(r,1);x=C.expand(d)
        @test all(k->isequal(getproperty(x,k),getproperty(r,k)),keys(r))
        @test C.record(C.from_record(C.record(d)))==C.record(d)
        @test d.context_id==1
    end
    @test C.encode(root(arm="no_turning_points"),1).E_arm==0
    @test C.encode(root(side="descending"),1).I_branch==2
    @test occursin("near singular",C.descriptor_text(C.encode(cases[5],1)))
    @test occursin("E2/I1",C.descriptor_text(C.encode(cases[1],1)))
    @test_throws ArgumentError C.encode(root(),0)
    @test_throws ArgumentError C.encode(merge(root(),(;root=65536)),1)
    @test_throws ArgumentError C.encode(root(E=.51),1)
    @test_throws ArgumentError C.encode(root(a=Inf),1)
    @test_throws ArgumentError C.encode(root(a=.2,b=-.1,w=.4),1)
    @test_throws ArgumentError C.encode(merge(root(),(;lambda1_imag=.2)),1)
    @test_throws KeyError C.encode(root(arm="unrecognized"),1)

    mktempdir() do dir
        d=C.encode(root(a=-.1,b=-.1,w=.4),1)
        contexts=[(;context_id=1,context="test/context",scope="intervention",case_id="test",
            B_E=2.,B_I=1.,axis="tau_ratio",value=.2,roots=1,source="source/context.toml",source_sha256=repeat("a",64))]
        C.write_gzip(joinpath(dir,"contexts.csv"),contexts)
        C.write_gzip(joinpath(dir,"roots.csv"),[C.record(d)])
        C.CSV.write(joinpath(dir,"chunks.csv"),[(;kind="contexts",file="contexts.csv.gz",rows=1),(;kind="roots",file="roots.csv.gz",rows=1)])
        C.write_toml(joinpath(dir,"parameters.toml"),Dict("cases"=>[Dict("id"=>"test","tau_ratio"=>4.4)]))
        meta=Dict("schema_version"=>1,"completed"=>true,"contexts"=>1,"roots"=>1,
            "parameter_registry_sha256"=>C.IOChecks.digest(joinpath(dir,"parameters.toml")),
            "E_arms"=>C.E_ARMS,"I_branches"=>C.I_BRANCHES,"stability_codes"=>C.STABILITY)
        C.write_toml(joinpath(dir,"catalog.toml"),meta);C.IOChecks.write_manifest(dir)
        catalog=C.read_catalog(dir)
        recovered=C.Descriptor[];C.foreach_descriptor(x->push!(recovered,x),catalog)
        @test C.record(only(recovered))==C.record(d)
        @test C.context(catalog,d).context=="test/context"
        p=C.context_parameters(catalog,d)
        @test p.parameters["tau_ratio"]==.2
        @test p.B_E==2. && p.B_I==1.
        @test catalog.parameters["test"]["tau_ratio"]==4.4
        # Reject semantic corruption even if a caller regenerates a checksum manifest.
        function replace_roots(rows)
            rm(joinpath(dir,"roots.csv.gz"));C.write_gzip(joinpath(dir,"roots.csv"),rows)
            C.CSV.write(joinpath(dir,"chunks.csv"),[(;kind="contexts",file="contexts.csv.gz",rows=1),
                (;kind="roots",file="roots.csv.gz",rows=length(rows))])
            C.IOChecks.write_manifest(dir)
        end
        replace_roots([merge(C.record(d),(;context_id=2))])
        @test_throws ErrorException C.read_catalog(dir)
        replace_roots([merge(C.record(d),(;E_arm=17))])
        @test_throws ArgumentError C.read_catalog(dir)
        replace_roots([C.record(d),C.record(d)])
        duplicate_error=try C.read_catalog(dir);nothing catch e;e end
        @test duplicate_error isa ErrorException && occursin("duplicate root identity",sprint(showerror,duplicate_error))
        replace_roots([C.record(d)])
        meta["schema_version"]=2;C.write_toml(joinpath(dir,"catalog.toml"),meta);C.IOChecks.write_manifest(dir)
        @test_throws ErrorException C.read_catalog(dir)
        meta["schema_version"]=1;C.write_toml(joinpath(dir,"catalog.toml"),meta);C.IOChecks.write_manifest(dir)
        write(joinpath(dir,"extra.txt"),"unlisted")
        @test_throws ErrorException C.read_catalog(dir)
    end
end

@testset "Continuous three-layer descriptions" begin
    C=EquilibriumCatalog
    for arm in values(C.E_ARMS), branch in values(C.I_BRANCHES), status in values(C.STABILITY)
        d=C.Descriptor(17,2,arm,branch,-0.,.5,-.02,-.1,0.,status,true)
        x=C.layers(d)
        @test x.identity==(context_id=d.context_id,root_id=d.root_id)
        @test isequal(x.activity,(E=d.E,I=d.I))
        @test all(isequal(getproperty(x.structure_dynamics,k),getproperty(d,k)) for k in keys(x.structure_dynamics))
        @test x.interpretation==(status=:deferred,label=nothing)
        @test occursin("biological interpretation deferred",C.layered_text(d))
        @test occursin("near singular",C.layered_text(d))
    end
    for (a,b,w) in ((-.1,-.1,.4),(0.,0.,.2),(.2,-.1,0.))
        d=C.Descriptor(1,1,2,1,.24,.27,a,b,w,0,false)
        x=C.layers(d)
        @test (x.structure_dynamics.lambda_max_real,x.structure_dynamics.lambda_min_real,x.structure_dynamics.omega)==(a,b,w)
        @test occursin(w==0 ? "real spectrum" : "complex spectrum",C.layered_text(d))
        @test x.interpretation.label===nothing
    end
end
