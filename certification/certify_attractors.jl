"""Validated fixed-parameter root coverage and conservative global attractor gate."""
module AttractorCertification

using FailureOfInhibition2025
using IntervalArithmetic
import TOML
import CSV
import SHA

const I = interval

_contains_zero(x) = inf(x) <= 0 <= sup(x)
_disjoint(left, right) = sup(left) < inf(right) || sup(right) < inf(left)
_strict_inside(inner, outer) = inf(outer) < inf(inner) && sup(inner) < sup(outer)
_require_guaranteed(values) = all(isguaranteed, values) ||
    throw(ArgumentError("interval calculation lost its enclosure guarantee"))

function _logistic(z)
    return inv(I(1) + exp(-z))
end

"""A separate interval evaluation of the documented point-model equations."""
function interval_field(model, state)
    model.drive isa NoDrive ||
        throw(ArgumentError("certification currently requires NoDrive"))
    E, Istate = state
    coupling = model.coupling
    B_E, B_I = FailureOfInhibition2025.drive_value(model.drive, 0.0)
    u_E = I(coupling.e_to_e) * E - I(coupling.i_to_e) * Istate + I(B_E)
    u_I = I(coupling.e_to_i) * E - I(coupling.i_to_i) * Istate + I(B_I)
    e_response = model.excitatory.response
    e_rate = _logistic(I(e_response.slope) * (u_E - I(e_response.threshold)))
    e_slope = I(e_response.slope) * e_rate * (I(1) - e_rate)
    i_response = model.inhibitory.response
    if i_response isa FailureOfInhibitionResponse
        midpoint = (I(i_response.onset_threshold) +
            I(i_response.failure_threshold)) / I(2)
        halfwidth = I(i_response.slope) *
            (I(i_response.failure_threshold) - I(i_response.onset_threshold)) / I(2)
        z = I(i_response.slope) * (u_I - midpoint)
        denominator = cosh(z) + cosh(halfwidth)
        i_rate = sinh(halfwidth) / denominator
        i_slope = -I(i_response.slope) * sinh(halfwidth) * sinh(z) /
            denominator^2
    else
        i_rate = _logistic(I(i_response.slope) *
            (u_I - I(i_response.threshold)))
        i_slope = I(i_response.slope) * i_rate * (I(1) - i_rate)
    end
    tau_E, tau_I = I(model.excitatory.timescale), I(model.inhibitory.timescale)
    f = [(-E + (I(1) - E) * e_rate) / tau_E,
         (-Istate + (I(1) - Istate) * i_rate) / tau_I]
    jac = reshape([
        (-I(1) - e_rate + (I(1) - E) * e_slope * I(coupling.e_to_e)) / tau_E,
        (I(1) - Istate) * i_slope * I(coupling.e_to_i) / tau_I,
        -(I(1) - E) * e_slope * I(coupling.i_to_e) / tau_E,
        (-I(1) - i_rate - (I(1) - Istate) * i_slope * I(coupling.i_to_i)) / tau_I,
    ], 2, 2)
    _require_guaranteed(f)
    _require_guaranteed(jac)
    return f, jac
end

function _krawczyk(field, jacobian, box)
    center = mid.(box)
    center_box = I.(center)
    f_center = field(center_box)
    J = jacobian(box)
    _require_guaranteed(f_center)
    _require_guaranteed(J)
    a, b, c, d = mid(J[1, 1]), mid(J[1, 2]), mid(J[2, 1]), mid(J[2, 2])
    determinant = a * d - b * c
    isfinite(determinant) && determinant != 0 || return nothing
    C = ((d / determinant, -b / determinant),
         (-c / determinant, a / determinant))
    all(isfinite, (C[1]..., C[2]...)) || return nothing
    correction = [I((row == column) ? 1 : 0) -
        sum(I(C[row][index]) * J[index, column] for index in 1:2)
        for row in 1:2, column in 1:2]
    displacement = [box[index] - center_box[index] for index in 1:2]
    K = [center_box[row] -
        sum(I(C[row][index]) * f_center[index] for index in 1:2) +
        sum(correction[row, index] * displacement[index] for index in 1:2)
        for row in 1:2]
    _require_guaranteed(K)
    norm_bound = maximum(sum(sup(abs(correction[row, column]))
        for column in 1:2) for row in 1:2)
    return (box=K, norm_bound=norm_bound)
end

function _subdivide(box)
    widths = [sup(value) - inf(value) for value in box]
    coordinate = widths[1] >= widths[2] ? 1 : 2
    lower, upper = inf(box[coordinate]), sup(box[coordinate])
    divider = lower + 0.51 * (upper - lower)
    lower < divider < upper || return nothing
    first_box, second_box = copy(box), copy(box)
    first_box[coordinate] = I(lower, divider)
    second_box[coordinate] = I(divider, upper)
    return first_box, second_box
end

function _root_classification(jacobian, enclosure)
    J = jacobian(enclosure)
    _require_guaranteed(J)
    trace = J[1, 1] + J[2, 2]
    determinant = J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]
    if sup(determinant) < 0
        return :saddle
    elseif inf(determinant) > 0 && sup(trace) < 0
        return :attracting
    elseif inf(determinant) > 0 && inf(trace) > 0
        return :repelling
    end
    return :unresolved
end

"""Cover a box by root-free cells and cells each certified to contain one root."""
function certify_roots(field, jacobian, domain; maximum_depth=36,
    maximum_boxes=100000)
    stack = [(copy(domain), 0)]
    roots, unresolved, excluded = NamedTuple[], NamedTuple[], NamedTuple[]
    processed = 0
    while !isempty(stack)
        box, depth = pop!(stack)
        processed += 1
        if processed > maximum_boxes
            push!(unresolved, (box=box, reason=:box_limit))
            append!(unresolved, [(box=item[1], reason=:box_limit)
                for item in stack])
            break
        end
        values = field(box)
        _require_guaranteed(values)
        if any(value -> !_contains_zero(value), values)
            push!(excluded, (box=box, reason=:range_exclusion))
            continue
        end
        test = _krawczyk(field, jacobian, box)
        if test !== nothing
            if any(index -> _disjoint(test.box[index], box[index]), 1:2)
                push!(excluded, (box=box, reason=:krawczyk_exclusion))
                continue
            end
            if test.norm_bound < 1 && all(index ->
                _strict_inside(test.box[index], box[index]), 1:2)
                push!(roots, (box=box, enclosure=test.box,
                    norm_bound=test.norm_bound,
                    classification=_root_classification(jacobian, test.box)))
                continue
            end
        end
        split = depth < maximum_depth ? _subdivide(box) : nothing
        if split === nothing
            push!(unresolved, (box=box, reason=:depth_or_precision_limit))
        else
            push!(stack, (split[2], depth + 1), (split[1], depth + 1))
        end
    end
    return (roots=roots, unresolved=unresolved,
        excluded_boxes=excluded, processed_boxes=processed,
        equilibrium_count=isempty(unresolved) ? length(roots) : nothing)
end

function _inward_boundary(field, domain)
    e0, e1 = inf(domain[1]), sup(domain[1])
    i0, i1 = inf(domain[2]), sup(domain[2])
    sides = (field([I(e0), I(i0, i1)]),
        field([I(e1), I(i0, i1)]),
        field([I(e0, e1), I(i0)]),
        field([I(e0, e1), I(i1)]))
    all(_require_guaranteed, sides)
    return inf(sides[1][1]) > 0 && sup(sides[2][1]) < 0 &&
        inf(sides[3][2]) > 0 && sup(sides[4][2]) < 0
end

function _negative_divergence(jacobian, domain; maximum_depth=20,
    maximum_boxes=100000)
    stack = [(copy(domain), 0)]
    processed = 0
    while !isempty(stack)
        box, depth = pop!(stack)
        processed += 1
        processed > maximum_boxes && return (status=:box_limit, processed=processed)
        J = jacobian(box)
        _require_guaranteed(J)
        divergence = J[1, 1] + J[2, 2]
        sup(divergence) < 0 && continue
        inf(divergence) >= 0 && return (status=:nonnegative_witness,
            processed=processed)
        split = depth < maximum_depth ? _subdivide(box) : nothing
        split === nothing && return (status=:unresolved, processed=processed)
        push!(stack, (split[2], depth + 1), (split[1], depth + 1))
    end
    return (status=:strict_negative, processed=processed)
end

"""
    certify_attractor_count(field, jacobian, domain; kwargs...)

Certify the number of minimal attracting limit sets only if equilibrium
coverage, hyperbolicity, inward boundary, and strict negative divergence all
pass. Otherwise report `:not_certified` with the completed proof obligations.
"""
function certify_attractor_count(field, jacobian, domain; maximum_depth=36,
    maximum_boxes=100000)
    roots = certify_roots(field, jacobian, domain; maximum_depth, maximum_boxes)
    inward = _inward_boundary(field, domain)
    divergence = any(root -> root.classification == :repelling, roots.roots) ?
        (status=:nonnegative_witness, processed=0) :
        _negative_divergence(jacobian, domain; maximum_depth, maximum_boxes)
    hyperbolic = all(root -> root.classification != :unresolved, roots.roots)
    complete = roots.equilibrium_count !== nothing && inward && hyperbolic &&
        divergence.status == :strict_negative
    return (status=complete ? :certified_exact : :not_certified,
        exact_count=complete ? count(root -> root.classification == :attracting,
            roots.roots) : nothing,
        certified_attracting_equilibria=count(root ->
            root.classification == :attracting, roots.roots),
        root_coverage=roots, inward_boundary=inward,
        divergence=divergence, hyperbolic=hyperbolic)
end

function _case_model(case)
    drive = NoDrive()
    return matched_point_models(
        excitatory=PopulationParameters(timescale=case["tau_e"],
            response=LogisticResponse(slope=case["slope_e"],
                threshold=case["theta_e"])),
        inhibitory_control=PopulationParameters(
            timescale=case["tau_e"] * case["tau_ratio"],
            response=LogisticResponse(slope=case["slope_i"],
                threshold=case["theta_on"])),
        failure_threshold=case["theta_off"],
        coupling=PointCoupling(e_to_e=case["e_to_e"],
            i_to_e=case["i_to_e"], e_to_i=case["e_to_i"],
            i_to_i=case["i_to_i"]), drive=drive).failure_of_inhibition
end

function run_case(config_path, case_name, output_dir; maximum_depth=36,
    maximum_boxes=100000)
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or empty"))
    config = TOML.parsefile(config_path)
    case = only(filter(item -> item["name"] == case_name, config["cases"]))
    model = _case_model(case)
    field(box) = first(interval_field(model, box))
    jacobian(box) = last(interval_field(model, box))
    result = certify_attractor_count(field, jacobian, [I(0.0, 1.0), I(0.0, 1.0)];
        maximum_depth, maximum_boxes)
    mkpath(output)
    source_hashes = Dict{String,String}()
    source_files = ["Project.toml", "Manifest.toml", "certification/Project.toml",
        "certification/Manifest.toml", "certification/certify_attractors.jl"]
    for (directory, _, files) in walkdir(joinpath(@__DIR__, "..", "src")), file in files
        endswith(file, ".jl") && push!(source_files,
            relpath(joinpath(directory, file), joinpath(@__DIR__, "..")))
    end
    for relative in sort!(source_files)
        source = joinpath(@__DIR__, "..", relative)
        target = joinpath(output, "source", relative)
        mkpath(dirname(target))
        cp(source, target)
        source_hashes[relative] = bytes2hex(SHA.sha256(read(target)))
    end
    cp(config_path, joinpath(output, "config.toml"))
    roots = [(box_E_lower=inf(root.box[1]), box_E_upper=sup(root.box[1]),
        box_I_lower=inf(root.box[2]), box_I_upper=sup(root.box[2]),
        E_lower=inf(root.enclosure[1]), E_upper=sup(root.enclosure[1]),
        I_lower=inf(root.enclosure[2]), I_upper=sup(root.enclosure[2]),
        norm_bound=root.norm_bound,
        classification=string(root.classification))
        for root in result.root_coverage.roots]
    if isempty(roots)
        CSV.write(joinpath(output, "roots.csv"),
            (box_E_lower=Float64[], box_E_upper=Float64[],
             box_I_lower=Float64[], box_I_upper=Float64[],
             E_lower=Float64[], E_upper=Float64[], I_lower=Float64[],
             I_upper=Float64[], norm_bound=Float64[], classification=String[]))
    else
        CSV.write(joinpath(output, "roots.csv"), roots)
    end
    unresolved = [(E_lower=inf(item.box[1]), E_upper=sup(item.box[1]),
        I_lower=inf(item.box[2]), I_upper=sup(item.box[2]),
        reason=string(item.reason)) for item in result.root_coverage.unresolved]
    if isempty(unresolved)
        CSV.write(joinpath(output, "unresolved_boxes.csv"),
            (E_lower=Float64[], E_upper=Float64[], I_lower=Float64[],
             I_upper=Float64[], reason=String[]))
    else
        CSV.write(joinpath(output, "unresolved_boxes.csv"), unresolved)
    end
    excluded = [(E_lower=inf(item.box[1]), E_upper=sup(item.box[1]),
        I_lower=inf(item.box[2]), I_upper=sup(item.box[2]),
        reason=string(item.reason)) for item in result.root_coverage.excluded_boxes]
    if isempty(excluded)
        CSV.write(joinpath(output, "excluded_boxes.csv"),
            (E_lower=Float64[], E_upper=Float64[], I_lower=Float64[],
             I_upper=Float64[], reason=String[]))
    else
        CSV.write(joinpath(output, "excluded_boxes.csv"), excluded)
    end
    metadata = Dict("case" => case_name, "status" => string(result.status),
        "exact_count" => result.exact_count === nothing ? "not_available" : result.exact_count,
        "certified_attracting_equilibria" => result.certified_attracting_equilibria,
        "equilibrium_count" => result.root_coverage.equilibrium_count === nothing ?
            "not_available" : result.root_coverage.equilibrium_count,
        "excluded_boxes" => length(result.root_coverage.excluded_boxes),
        "processed_boxes" => result.root_coverage.processed_boxes,
        "unresolved_boxes" => length(result.root_coverage.unresolved),
        "inward_boundary" => result.inward_boundary,
        "divergence_status" => string(result.divergence.status),
        "hyperbolic" => result.hyperbolic,
        "maximum_depth" => maximum_depth, "maximum_boxes" => maximum_boxes,
        "source_sha256" => source_hashes,
        "config_sha256" => bytes2hex(SHA.sha256(read(joinpath(output, "config.toml")))),
        "julia_version" => string(VERSION),
        "model_parameter_convention" => "constructed Float64 model values treated as exact binary inputs",
        "replay_from_artifact_directory" =>
            "julia --project=source/certification source/certification/certify_attractors.jl --config config.toml --case $case_name --output replay --maximum-depth $maximum_depth --maximum-boxes $maximum_boxes")
    open(joinpath(output, "metadata.toml"), "w") do io
        TOML.print(io, metadata; sorted=true)
    end
    checksums = Dict{String,String}()
    for (directory, _, files) in walkdir(output), name in files
        relative = relpath(joinpath(directory, name), output)
        checksums[relative] = bytes2hex(SHA.sha256(read(joinpath(directory, name))))
    end
    open(joinpath(output, "checksums.toml"), "w") do io
        TOML.print(io, Dict("algorithm" => "SHA-256", "files" => checksums); sorted=true)
    end
    return result
end

function main(args=ARGS)
    options = Dict{String,String}()
    index = 1
    while index <= length(args)
        key = args[index]
        key in ("--config", "--case", "--output", "--maximum-depth",
            "--maximum-boxes") && index < length(args) && !haskey(options, key) ||
            throw(ArgumentError("unknown, duplicate, or incomplete option: $key"))
        options[key] = args[index + 1]
        index += 2
    end
    haskey(options, "--output") || throw(ArgumentError("--output is required"))
    config = get(options, "--config", joinpath(@__DIR__, "..", "experiments",
        "basin_rescue.toml"))
    depth = parse(Int, get(options, "--maximum-depth", "36"))
    boxes = parse(Int, get(options, "--maximum-boxes", "100000"))
    depth >= 0 && boxes > 0 || throw(ArgumentError("limits must be nonnegative"))
    result = run_case(config, get(options, "--case", "figure3"), options["--output"];
        maximum_depth=depth, maximum_boxes=boxes)
    println("Attractor status: $(result.status); certified attracting equilibria: " *
        "$(result.certified_attracting_equilibria); exact count: $(result.exact_count)")
    return 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(AttractorCertification.main())
end
