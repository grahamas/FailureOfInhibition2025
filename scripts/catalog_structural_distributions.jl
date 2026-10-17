"""Conditional occupancy distributions from catalog geometry, without biological cutoffs."""
module CatalogStructuralDistributions
include("equilibrium_catalog.jl")
include("equilibrium_nomenclature.jl")
const C=EquilibriumCatalog
const N=EquilibriumNomenclature
const COHORTS=("screen","detail_geometry")
const QS=(0.,.05,.25,.5,.75,.95,1.)
const EDGES=collect(range(0.,.5;length=51))
eligible(d)=d.stability==1 && !d.near_singular

"""Exact observed-value quantiles and histogram counts, including both occupancy endpoints."""
function distribution(values)
    all(x->isfinite(x) && 0<=x<=.5,values) || throw(ArgumentError("invalid occupancy"))
    xs=sort(collect(values));n=length(xs);counts=zeros(Int,50)
    for x in xs
        counts[clamp(searchsortedlast(EDGES,x),1,50)]+=1
    end
    (;n,quantiles=isempty(xs) ? fill(missing,length(QS)) : [N.observed_quantile(xs,q) for q in QS],
      counts,zeros=count(iszero,xs),ceiling=count(==(.5),xs))
end

"""Exact right-continuous empirical CDF coordinates; tied values make a single jump.

The plotted vertical segment at zero shows the atom at zero. The distribution
is normalized within this group, not across all catalog records.
"""
function ecdf_steps(values)
    all(x->isfinite(x) && 0<=x<=.5,values) || throw(ArgumentError("invalid occupancy"))
    isempty(values) && return (Float64[],Float64[])
    xs=sort(collect(values));x=[0.];y=[0.];n=length(xs);i=1
    while i<=n
        j=searchsortedlast(xs,xs[i])
        append!(x,(xs[i],xs[i]));append!(y,((i-1)/n,j/n));i=j+1
    end
    push!(x,.5);push!(y,1.);x,y
end

"""Joint E–I counts with E on the first axis; exact 0.5 belongs in the last bin."""
function joint_histogram(E,I)
    length(E)==length(I) || throw(DimensionMismatch("E and I lengths differ"))
    all(x->isfinite(x) && 0<=x<=.5,E) && all(x->isfinite(x) && 0<=x<=.5,I) || throw(ArgumentError("invalid occupancy"))
    edges=collect(range(0.,.5;length=101));counts=zeros(Int,100,100)
    for (e,i) in zip(E,I)
        counts[clamp(searchsortedlast(edges,e),1,100),clamp(searchsortedlast(edges,i),1,100)]+=1
    end
    counts
end

function load_groups(catalog)
    groups=Dict{Tuple{String,Int,Int},Vector{C.Descriptor}}()
    C.foreach_descriptor(catalog) do d
        scope=C.context(catalog,d).scope
        scope in COHORTS || return
        # -1 denotes a marginal over the other structural dimension.
        for key in ((scope,Int(d.E_arm),Int(d.I_branch)),(scope,Int(d.E_arm),-1),(scope,-1,Int(d.I_branch)),(scope,-1,-1))
            push!(get!(groups,key,C.Descriptor[]),d)
        end
    end
    groups
end

function build(directory,out)
    catalog=C.read_catalog(directory);groups=load_groups(catalog)
    ispath(out) && error("output exists");mkpath(out)
    summary=NamedTuple[];histograms=NamedTuple[];stability=NamedTuple[];joint=NamedTuple[]
    for ((scope,arm,branch),ds) in sort(collect(groups);by=first)
        stable=filter(eligible,ds)
        push!(stability,(;scope,E_arm=arm,I_branch=branch,n=length(ds),
            attracting=count(d->d.stability==1,ds),saddle=count(d->d.stability==2,ds),
            repelling=count(d->d.stability==3,ds),unresolved=count(d->d.stability==0,ds),
            near_singular=count(d->d.near_singular,ds),eligible=length(stable)))
        for (subset,records) in (("all",ds),("attracting_nonsingular",stable))
            h=joint_histogram(getproperty.(records,:E),getproperty.(records,:I))
            for index in findall(>(0),h)
                push!(joint,(;scope,E_arm=arm,I_branch=branch,subset,E_bin=index[1],I_bin=index[2],count=h[index],n=length(records)))
            end
        end
        for (subset,records) in (("all",ds),("attracting_nonsingular",stable)), field in (:E,:I)
            s=distribution(getproperty.(records,field))
            push!(summary,(;scope,E_arm=arm,I_branch=branch,subset,variable=String(field),n=s.n,
                minimum=s.quantiles[1],q05=s.quantiles[2],q25=s.quantiles[3],median=s.quantiles[4],
                q75=s.quantiles[5],q95=s.quantiles[6],maximum=s.quantiles[7],exact_zero=s.zeros,exact_ceiling=s.ceiling))
            for (i,count) in enumerate(s.counts)
                push!(histograms,(;scope,E_arm=arm,I_branch=branch,subset,variable=String(field),
                    bin=i,lo=EDGES[i],hi=EDGES[i+1],count,n=s.n))
            end
        end
    end
    C.CSV.write(joinpath(out,"quantiles.csv"),summary)
    C.write_gzip(joinpath(out,"histograms.csv"),histograms)
    C.CSV.write(joinpath(out,"stability.csv"),stability)
    C.write_gzip(joinpath(out,"joint_histograms.csv"),joint)
    C.write_toml(joinpath(out,"study.toml"),Dict("completed"=>true,"schema_version"=>1,
        "catalog_manifest_sha256"=>C.IOChecks.digest(joinpath(directory,"checksums.toml")),
        "analysis_sha256"=>C.IOChecks.digest(@__FILE__),
        "quantile_helper_sha256"=>C.IOChecks.digest(joinpath(@__DIR__,"equilibrium_nomenclature.jl")),
        "catalog_reader_sha256"=>C.IOChecks.digest(joinpath(@__DIR__,"equilibrium_catalog.jl")),
        "cohorts"=>collect(COHORTS),"marginal_code"=>-1,"quantiles"=>collect(QS),
        "quantile_convention"=>"inverse empirical CDF: observed value at max(1,ceil(q*n)); no interpolation",
        "histogram_convention"=>"50 equal-width bins on [0,0.5]; left-inclusive, right-exclusive except final bin includes 0.5",
        "joint_histogram_convention"=>"100 by 100 bins, width 0.005, E first axis; sparse positive counts only, empty bins implicit zero; final bins include 0.5",
        "normalization"=>"within structural category and subset; each root record has equal weight",
        "limits"=>"Adaptive, overlapping cohorts. Not physiological prevalence, basin weights, or independent branches. CompletenessNotCertified."))
    C.IOChecks.write_manifest(out)
    println("Wrote $(length(summary)) distributions and $(length(stability)) structural groups")
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: catalog_structural_distributions.jl CATALOG NEW_OUTPUT")
    CatalogStructuralDistributions.build(abspath.(ARGS)...)
end
