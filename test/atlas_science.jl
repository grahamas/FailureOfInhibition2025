using Test
using FailureOfInhibition2025
using LinearAlgebra
module AtlasScienceHarness
include(joinpath(@__DIR__,"..","scripts","atlas","core.jl"))
include(joinpath(@__DIR__,"..","scripts","atlas","science.jl"))
end
using .AtlasScienceHarness
const ASH=AtlasScienceHarness
@testset "First-principles atlas coordinates" begin
    p=ASH.discovery_job(3,"parameter",false)["params"]
    x=[0.2,0.3];f=zeros(2);g=zeros(2)
    point_rhs!(f,x,ASH.reduced_model(p),0.0)
    point_rhs!(g,x,ASH.physical_model(p;slope_e=2.3,slope_i=4.7,tau_e=7.8),0.0)
    @test f≈7.8g atol=1e-14
    for v in (-8.0,-1.0,0.0,2.0,8.0)
        chart=ASH.equilibrium_chart(12.0,8.0,8.0,2.0,0.5,v,2.0)
        @test maximum(abs,ASH.balance_value(chart["params"],chart["state"]))<1e-14
    end
    chart=ASH.fold_chart(8.,8.,0.,-log(2.),0.,log(9.))
    @test chart["fold_residual"]<1e-12
    bt=ASH.analytic_seed();K=ASH.balance_jacobian(bt["params"],bt["state"])
    @test abs(det(K))<1e-12
    @test abs(tr(Diagonal([1.0,1/bt["params"]["r"]])*K))<1e-12
    @test_throws ArgumentError ASH.reduced_model(merge(p,Dict("h"=>0.0)))
end
@testset "Disjoint adaptive shells and boundaries" begin
    for chart in ("parameter","equilibrium"), k in 1:6, i in 1:10, mask in (0,1,15)
        sample=ASH.shell_sample(k,i;chart=chart,mask=mask);z=sample["chart_coordinates"]
        @test ASH.in_window(z,k)
        @test k==1 || !ASH.in_window(z,k-1)
        for j in 1:4
            mask & (1<<(j-1))!=0 && @test z[j]==0.0
        end
        @test sample==ASH.shell_sample(k,i;chart=chart,mask=mask)
    end
end
@testset "Analytic ratio boundaries" begin
    distant=ASH.ratio_partition([1e-8 1.; -1. -1.])
    @test length(distant["boundaries"])==3
    @test last(distant["boundaries"])≈4e16 rtol=1e-7
    @test last(distant["intervals"])["spectrum"]=="real"
    @test ASH.ratio_partition([1e-12 -2.;3. -4.])["hopf_candidate"]
    @test_throws ArgumentError ASH.ratio_partition([2e-154 -1.;1. -1.])

    for K in ([-1.0 -2;3 -4], [1.0 -2;3 -4], [1.0 2;3 -4], [0.0 0;0 -1])
        partition=ASH.ratio_partition(K)
        for interval in partition["intervals"]
            r=interval["probe"];ev=eigvals(Diagonal([1.0,1/r])*K)
            label=interval["stability"]
            label=="attracting" && @test maximum(real.(ev))<0
            label=="repelling" && @test minimum(real.(ev))>0
            label=="saddle" && @test prod(real.(ev))<0
        end
    end
end
@testset "Discovery retains attempts and certification scope" begin
    job=ASH.discovery_job(1,"parameter",false)
    result=ASH.execute_job(job,Dict{String,Any}("seed_grid"=>3))
    @test result["status"]=="completed"
    point=only(result["evidence"]["points"])
    @test length(point["attempts"])==9
    @test point["completeness"]=="CompletenessNotCertified"
    cert=ASH.execute_job(Dict{String,Any}("kind"=>"certification","params"=>job["params"]),Dict{String,Any}("certification_boxes"=>0))
    @test cert["status"]=="unresolved"
    @test only(cert["evidence"]["certifications"])["scope"]=="fixed_parameter_equilibria_only"
end
@testset "Per-root chart geometry and retained chart seeds" begin
    config=Dict{String,Any}("seed_grid"=>3)
    for (u,label) in ((-4.0,"left"),(-log(2.0),"middle"),(4.0,"right"))
        chart=ASH.equilibrium_chart(12.,1.,1.,0.,u,0.,2.)
        point=ASH.atlas_point(chart["params"],config;extra_seeds=[chart["state"]])
        index=argmin([norm(r["state"]-chart["state"]) for r in point["roots"]])
        @test point["roots"][index]["excitatory_chart_branch"]==label
    end
    for (v,label) in ((0.,"left"),(3.,"middle"),(15.,"right"))
        chart=ASH.equilibrium_chart(1.,1.,1.,100.,0.,v,2.)
        point=ASH.atlas_point(chart["params"],config;extra_seeds=[chart["state"]])
        index=argmin([norm(r["state"]-chart["state"]) for r in point["roots"]])
        @test point["roots"][index]["inhibitory_self_chart_branch"]==label
    end
    job=ASH.discovery_job(1,"equilibrium",false)
    result=ASH.execute_job(job,config)
    @test length(only(result["evidence"]["points"])["attempts"])==10
    @test !isempty(result["evidence"]["events"])
    @test all(j->!get(j,"fixture",false),ASH.initial_jobs(Dict("mode"=>"run")))
end
@testset "Event adapters preserve saved scientific seeds" begin
    config=Dict{String,Any}("seed_grid"=>2)
    chart=ASH.equilibrium_chart(12.,0.,0.,0.,-log(2.),0.,log(9.))
    p,x=chart["params"],chart["state"]
    plain=ASH.atlas_point(p,config)
    @test minimum(norm(root["state"]-x) for root in plain["roots"])>0.1
    section=Dict{String,Any}("section_samples"=>[Dict("parameters"=>p,"equilibria"=>[Dict("state"=>x)])])
    point=only(ASH.atlas_anchor_points(section,"section",Dict("id"=>"regression"),config))
    @test length(point["attempts"])==5
    @test first(point["attempts"])["seed"]==x
    @test minimum(norm(root["state"]-x) for root in point["roots"])<1e-7
    cycle=Dict{String,Any}("validated"=>true,"parameters"=>p,"hopf_state"=>x)
    anchor=only(ASH.atlas_anchor_points(cycle,"periodic",Dict("id"=>"cycle"),config))
    @test first(anchor["attempts"])["seed"]==x
    job=Dict{String,Any}("kind"=>"section","parameters"=>p)
    normalized=ASH.atlas_normalize_job(job)
    @test normalized["params"]==p
    @test normalized["queue"]=="events"
    @test !haskey(job,"params")
    @test_throws ArgumentError ASH.atlas_normalize_job(merge(job,Dict("params"=>merge(p,Dict("A"=>1.0)))))
end
