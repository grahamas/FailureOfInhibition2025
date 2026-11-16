# Bounded calibration against immutable training evidence. No discovery,
# shooting, continuation campaign, or heldout records are used here.
function atlas_calibration_policy()
    Dict{String,Any}("policy_version"=>ATLAS_NUMERICAL_POLICY_VERSION,"scope"=>"training-only",
        "balance_atol"=>1e-9,"event_atol"=>1e-9,
        "stage_caps"=>Dict("targeted"=>Dict("balance_atol"=>1e-11,"event_atol"=>1e-9)),
        "calibration_storage_cap_bytes"=>256*1024^2,
        "coefficient_floor"=>1e-5,"coefficient_rtol"=>0.02,
        "derivative_reference_bits"=>ATLAS_DERIVATIVE_REFERENCE_BITS,
        "derivative_reference_step"=>ATLAS_DERIVATIVE_REFERENCE_STEP,
        "derivative_reference_refinement_step"=>ATLAS_DERIVATIVE_REFERENCE_STEP/2,
        "tensor_relative_budget"=>2e-12,"tensor_absolute_budget"=>2e-14,
        "reference_relative_budget"=>1e-23,"reference_absolute_budget"=>1e-25,
        "package_equation_relative_budget"=>2e-13,"package_equation_absolute_budget"=>2e-14,
        "error_budget"=>Dict("separate_error_categories"=>true,
            "balance"=>"Unscaled dimensionless balance residual, independently recomputed; cap 1e-9; stricter configured caps remain binding",
            "event"=>"Original-time event residual recomputed with analytic state derivatives; cap 1e-9, no increase from failed targeted packet",
            "coefficient"=>"Primary analytic versus 256-bit direct-equation FD reference at two steps; nonzero magnitude >1e-5, same sign, <=2 percent relative discrepancy",
            "tensor"=>"2e-12 relative +2e-14 absolute budget is over 100 Float64 roundoffs and independent of event residual caps; reference refinement <=1e-23 relative +1e-25 absolute",
            "matching"=>"Root/event identity matching is separate from residual/coefficient acceptance; no matching tolerance chosen by calibration",
            "conditioning"=>"Report scaled singular values and local residual-to-coordinate estimate; these estimates are diagnostics, not rigorous upper bounds",
            "threshold_decision"=>"Retain 1e-9 event cap; remove observed coarse finite-difference correction bias using analytic derivatives"),
        "acceptance_scope"=>"Numerical training calibration only; independent scientific review and fresh validation remain separately required",
        "completeness"=>"CompletenessNotCertified")
end

function atlas_calibration_tuple(p,x,policy)
    diagnostics=atlas_derivative_diagnostics(p,x)
    H,C=atlas_reduced_tensor_arrays(p,x)
    Hr,Cr=atlas_reference_tensor_arrays(p,x;step=ATLAS_DERIVATIVE_REFERENCE_STEP/2)
    tensor_pass=all(diagnostics[key]<=policy["tensor_absolute_budget"]+
        policy["tensor_relative_budget"]*Float64(maximum(abs,a)) for (key,a) in
        (("hessian_maximum_absolute_difference",Hr),("third_maximum_absolute_difference",Cr)))
    reference_pass=all(diagnostics[key]<=policy["reference_absolute_budget"]+
        policy["reference_relative_budget"]*Float64(maximum(abs,a)) for (key,a) in
        (("hessian_reference_refinement_difference",Hr),("third_reference_refinement_difference",Cr)))
    equation_pass=all(diagnostics[difference]<=policy["package_equation_absolute_budget"]+
        policy["package_equation_relative_budget"]*diagnostics[scale] for (difference,scale) in
        (("package_rhs_reference_difference","package_rhs_reference_scale"),
        ("package_jacobian_reference_difference","package_jacobian_reference_scale")))
    symmetry=max(maximum(abs.(H.-permutedims(H,(1,3,2)))),
        maximum(abs.(C.-permutedims(C,(1,3,2,4)))),maximum(abs.(C.-permutedims(C,(1,4,3,2)))))
    symmetry_pass=symmetry<=policy["tensor_absolute_budget"]+
        policy["tensor_relative_budget"]*max(maximum(abs,H),maximum(abs,C))
    merge(diagnostics,Dict("passed"=>tensor_pass&&reference_pass&&equation_pass&&symmetry_pass,
        "tensor_budget_passed"=>tensor_pass,"reference_refinement_passed"=>reference_pass,
        "package_equation_budget_passed"=>equation_pass,"tensor_symmetry_passed"=>symmetry_pass,
        "maximum_tensor_permutation_difference"=>symmetry,"parameters"=>p,"state"=>x))
end

function atlas_calibration_controls(policy;targeted_source=nothing)
    config=Dict{String,Any}("balance_atol"=>1e-11,"event_atol"=>policy["event_atol"])
    controls=Dict{String,Any}[]
    cusp=merge(equilibrium_chart(8.,0.,0.,0.,-log(2.),0.,log(9.)),Dict("kind"=>"nexus","event_type"=>"cusp"))
    for (name,job) in (("exact_cusp",deepcopy(cusp)),("perturbed_cusp",deepcopy(cusp)),
            ("exact_bt",analytic_seed()),("perturbed_bt",deepcopy(analytic_seed())))
        startswith(name,"perturbed") && (job["state"]+= [.003,-.002])
        evidence=atlas_nexus_job(job,config)
        target=name in ("exact_cusp","perturbed_cusp") ? 0.25 : analytic_seed()["state"][1]
        push!(controls,Dict("name"=>name,"passed"=>evidence["validated"]&&abs(evidence["state"][1]-target)<=1e-8,
            "evidence"=>evidence))
    end
    hp=equilibrium_chart(12.,8.,8.,0.,-log(2.),0.,log(9.);r=2.1)
    p,x=hp["params"],hp["state"]
    l1,omega,_=atlas_hopf_coefficient(y->atlas_jac(p,y),x;tensors=atlas_reduced_tensors(p,x))
    lr,_,_=atlas_hopf_coefficient(y->atlas_jac(p,y),x;tensors=atlas_reference_tensors(p,x))
    lf,_,_=atlas_hopf_coefficient(y->atlas_jac(p,y),x;
        tensors=atlas_reference_tensors(p,x;step=ATLAS_DERIVATIVE_REFERENCE_STEP/2))
    push!(controls,Dict("name"=>"actual_model_hopf","passed"=>l1<0&&omega>0&&
        atlas_coefficient_resolved(l1,lf)&&atlas_coefficient_resolved(lr,lf),
        "coefficient"=>l1,"reference_coefficient"=>lr,"refined_reference_coefficient"=>lf,
        "omega"=>omega,"derivative_diagnostics"=>atlas_derivative_diagnostics(p,x)))
    quintic=x->[100x[1]^4 0.;0. -1.]
    c=atlas_nexus_coefficients(quintic,zeros(2),"cusp")["cubic"]
    f=atlas_nexus_coefficients(quintic,zeros(2),"cusp";step=5e-4)["cubic"]
    push!(controls,Dict("name"=>"negative_quintic_cusp","passed"=>!atlas_coefficient_resolved(c,f),
        "coarse_coefficient"=>c,"refined_coefficient"=>f))
    bt_degenerate=x->[0. 1.;0. 0.]
    c=atlas_nexus_coefficients(bt_degenerate,zeros(2),"bt")
    push!(controls,Dict("name"=>"negative_degenerate_bt","passed"=>
        !atlas_coefficient_resolved(c["quadratic"],c["quadratic"])&&
        !atlas_coefficient_resolved(c["mixed"],c["mixed"]),"coefficients"=>c))
    J=x->[0. -1.;1. 0.]
    coefficient,_,_=atlas_hopf_coefficient(J,zeros(2))
    push!(controls,Dict("name"=>"negative_zero_lyapunov_hopf","passed"=>
        !atlas_coefficient_resolved(coefficient,coefficient),"coefficient"=>coefficient))
    for (name,p,x) in (("narrow_inhibitory_width",Dict(zip(ATLAS_COORDINATES,
            (128.,64.,32.,16.,30.,4.,1e-5,0.01))),[.17,.31]),
            ("large_mixed_coupling",Dict(zip(ATLAS_COORDINATES,
            (64.,128.,128.,64.,-25.,-16.,32.,100.))),[.49,.01]))
        evidence=atlas_calibration_tuple(p,x,policy)
        push!(controls,Dict("name"=>name,"passed"=>evidence["passed"],"evidence"=>evidence))
    end
    if targeted_source!==nothing
        old=TOML.parsefile(joinpath(targeted_source,"results","job-000002.toml"))
        saved=only(filter(row->get(row,"classification","")=="candidate_cusp",old["evidence"]["events"]))
        p,x=saved["parameters"],saved["state"]
        c=atlas_nexus_coefficients(y->atlas_jac(p,y),x,"cusp";tensors=atlas_reduced_tensors(p,x))
        push!(controls,Dict("name"=>"failed_targeted_cusp_bias_reproduced","passed"=>
            abs(c["quadratic"])>policy["event_atol"],"analytic_quadratic"=>c["quadratic"],
            "old_refined_quadratic"=>saved["coefficients"]["quadratic"],
            "old_coarse_saved_event_residual"=>saved["event_residual"],
            "reason"=>"Old coarse FD corrector tuple remains outside analytic event cap; packet is immutable"))
    end
    controls
end

"""Calibrate only frozen original training tuples and bounded controls.

Writes a new root that must be retained, byte-counted and frozen with the campaign.
Readiness reports numerical calibration, not independent review or fresh validation.
"""
function atlas_calibrate_numerics(source,output;targeted_source=nothing)
    ispath(output)&&throw(ArgumentError("calibration output already exists"))
    source=realpath(source);output=abspath(output)
    policy=atlas_calibration_policy()
    state=TOML.parsefile(joinpath(source,"state.toml"))
    jobs=filter(job->get(job["payload"],"pilot_role","")=="training",state["jobs"])
    length(jobs)==512 || throw(ArgumentError("original pilot must contain exactly 512 training jobs"))
    provenance=Dict{String,Any}[]
    function provenance!(root,path)
        file=joinpath(root,path)
        push!(provenance,Dict("source_root"=>root,"path"=>path,"sha256"=>bytes2hex(sha256(read(file))),"bytes"=>filesize(file)))
    end
    provenance!(source,"state.toml");provenance!(source,"config.toml")
    frozen_core=joinpath(source,"source","scripts","atlas","core.jl")
    bytes2hex(sha256(read(frozen_core)))==bytes2hex(sha256(read(joinpath(@__DIR__,"core.jl")))) ||
        throw(ArgumentError("current sampling/chart reconstruction differs from frozen training source"))
    provenance!(source,"source/scripts/atlas/core.jl")
    rows=Dict{String,Any}[]
    for job in jobs
        path=job["result_file"];file=joinpath(source,path)
        startswith(realpath(file),source*"/") || throw(ArgumentError("training result outside source packet"))
        bytes2hex(sha256(read(file)))==job["result_hash"] || throw(ArgumentError("training source result hash mismatch"))
        provenance!(source,path)
        points=get(TOML.parsefile(file)["evidence"],"points",Any[])
        job["status"] in ("completed","failed") || throw(ArgumentError("unsupported training source status"))
        reconstructed=shell_sample(job["payload"]["pilot_shell"],job["payload"]["pilot_sample_index"];
            chart=job["payload"]["pilot_chart"],mask=job["payload"]["pilot_mask"])["params"]
        length(points)<=1 || throw(ArgumentError("at most one frozen training point required"))
        job["status"]=="completed" && isempty(points) && throw(ArgumentError("completed training point is missing"))
        p=isempty(points) ? reconstructed : only(points)["parameters"]
        p==reconstructed || throw(ArgumentError("frozen training parameters differ from manifest reconstruction"))
        roots=isempty(points) ? Any[] : only(points)["roots"]
        states=isempty(roots) ? [[.25,.25]] : [root["state"] for root in roots]
        for (index,x) in enumerate(states)
            row=atlas_calibration_tuple(p,Float64.(x),policy)
            merge!(row,Dict("source_job_id"=>job["id"],"source_result_hash"=>job["result_hash"],
                "state_selection"=>isempty(roots) ? "predeclared_domain_center_no_saved_root" : "saved_root",
                "original_job_status"=>job["status"],"original_failure_retained"=>job["status"]=="failed",
                "saved_root_index"=>isempty(roots) ? 0 : index,"saved_root_count"=>length(roots),
                "chart"=>job["payload"]["pilot_chart"],"shell"=>job["payload"]["pilot_shell"]))
            push!(rows,row)
        end
    end
    targeted_source!==nothing && provenance!(realpath(targeted_source),"results/job-000002.toml")
    controls=atlas_calibration_controls(policy;targeted_source=targeted_source)
    ready=all(row["passed"] for row in rows)&&all(row["passed"] for row in controls)
    calibration=Dict{String,Any}("policy_version"=>ATLAS_NUMERICAL_POLICY_VERSION,"readiness"=>ready,
        "scope"=>"original_512_training_parameter_tuples_and_declared_controls_only",
        "training_parameter_tuple_count"=>512,"derivative_evaluation_tuple_count"=>length(rows),
        "original_completed_training_jobs"=>count(job->job["status"]=="completed",jobs),
        "original_failed_training_jobs"=>count(job->job["status"]=="failed",jobs),
        "training_failures"=>count(row->!row["passed"],rows),"control_failures"=>count(row->!row["passed"],controls),
        "tuples"=>rows,"controls"=>controls,"completeness"=>"CompletenessNotCertified")
    mkpath(output)
    storage=storage_policy(policy["calibration_storage_cap_bytes"])
    budget=StorageBudget(output,storage["cap_bytes"],storage["checkpoint_reserve_bytes"],
        storage["report_reserve_bytes"],storage["result_reserve_bytes"],storage["monitor_headroom_bytes"],
        storage["file_limit_bytes"],0)
    storage_atomic_write(joinpath(output,".storage.toml"),io->TOML.print(io,storage;sorted=true);
        budget=budget,category="checkpoint")
    implementation=Dict("files"=>[Dict("path"=>path,"sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"..","..",path)))))
        for path in ("scripts/atlas/core.jl","scripts/atlas/events.jl","scripts/atlas/derivatives.jl",
            "scripts/atlas/calibration.jl","scripts/atlas/targeted_events.jl",
            "src/responses.jl","src/point_model.jl")])
    for (name,data) in (("calibration.toml",calibration),("source_manifest.toml",Dict("sources"=>provenance)),
            ("implementation_manifest.toml",implementation))
        storage_atomic_write(joinpath(output,name),io->TOML.print(io,data;sorted=true);budget=budget,category="report")
    end
    merge!(policy,Dict("readiness"=>ready,"calibration_root"=>output,"calibration_sources"=>[output],
        "calibration_files"=>[Dict("path"=>name,"sha256"=>bytes2hex(sha256(read(joinpath(output,name)))) ,
            "bytes"=>filesize(joinpath(output,name))) for name in ("calibration.toml","source_manifest.toml","implementation_manifest.toml")]))
    storage_atomic_write(joinpath(output,"policy.toml"),io->TOML.print(io,policy;sorted=true);budget=budget,category="report")
    policy
end

"""Validate supported numerical choices and their complete calibration binding.

Independent scientific review is a separate parent-owned artifact/gate. This
checks numerical readiness and exact source/output bytes, never review approval.
"""
function atlas_validate_numerical_policy(policy)
    policy isa AbstractDict || throw(ArgumentError("numerical policy must be a table"))
    baseline=atlas_calibration_policy()
    all(haskey(policy,key)&&policy[key]==value for (key,value) in baseline) ||
        throw(ArgumentError("unsupported or altered numerical policy choices"))
    get(policy,"readiness",false)===true || throw(ArgumentError("numerical calibration is not ready"))
    root=get(policy,"calibration_root","")
    root isa String && isabspath(root)&&isdir(root)&&!islink(root) || throw(ArgumentError("calibration root must be an existing absolute directory"))
    realpath(root)==root || throw(ArgumentError("calibration root must be canonical"))
    get(policy,"calibration_sources",Any[])==[root] || throw(ArgumentError("calibration directory binding mismatch"))
    files=get(policy,"calibration_files",Any[])
    required=Set(("calibration.toml","source_manifest.toml","implementation_manifest.toml"))
    length(files)==length(required)&&Set(get(row,"path","") for row in files)==required ||
        throw(ArgumentError("complete calibration evidence inventory required"))
    for row in files
        path=joinpath(root,row["path"])
        isfile(path)&&!islink(path) || throw(ArgumentError("calibration evidence missing or symlinked"))
        row["bytes"]==filesize(path)&&row["sha256"]==bytes2hex(sha256(read(path))) ||
            throw(ArgumentError("calibration evidence bytes changed"))
    end
    calibration=TOML.parsefile(joinpath(root,"calibration.toml"))
    calibration["policy_version"]==ATLAS_NUMERICAL_POLICY_VERSION&&calibration["readiness"]===true&&
        calibration["training_parameter_tuple_count"]==512&&calibration["training_failures"]==0&&
        calibration["control_failures"]==0 || throw(ArgumentError("incomplete numerical calibration"))
    rows=calibration["tuples"]
    length(Set(row["source_job_id"] for row in rows))==512&&all(row["passed"]===true for row in rows) ||
        throw(ArgumentError("calibration does not cover all original training parameters"))
    expected_controls=Set(("exact_cusp","perturbed_cusp","exact_bt","perturbed_bt","actual_model_hopf",
        "negative_quintic_cusp","negative_degenerate_bt","negative_zero_lyapunov_hopf",
        "narrow_inhibitory_width","large_mixed_coupling"))
    controls=calibration["controls"]
    issubset(expected_controls,Set(row["name"] for row in controls))&&all(row["passed"]===true for row in controls) ||
        throw(ArgumentError("calibration controls did not all pass"))
    for row in TOML.parsefile(joinpath(root,"source_manifest.toml"))["sources"]
        path=joinpath(row["source_root"],row["path"])
        isfile(path)&&!islink(path)&&realpath(path)==abspath(path)&&
            filesize(path)==row["bytes"]&&bytes2hex(sha256(read(path)))==row["sha256"] ||
            throw(ArgumentError("immutable calibration input changed"))
    end
    for row in TOML.parsefile(joinpath(root,"implementation_manifest.toml"))["files"]
        path=joinpath(@__DIR__,"..","..",row["path"])
        isfile(path)&&bytes2hex(sha256(read(path)))==row["sha256"] ||
            throw(ArgumentError("calibrated numerical implementation changed"))
    end
    policy
end
