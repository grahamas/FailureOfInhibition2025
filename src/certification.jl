import IntervalArithmetic

const _CIA = IntervalArithmetic

"""Resource bounds for validated fixed-parameter equilibrium coverage."""
struct CertificationOptions
    maximum_boxes::Int
    maximum_depth::Int
    maximum_evaluations::Int
    maximum_records::Int
    time_limit::Float64
    maximum_precision::Int
    function CertificationOptions(; maximum_boxes=100_000, maximum_depth=64,
        maximum_evaluations=1_000_000, maximum_records=200_000,
        time_limit=60.0, maximum_precision=1024)
        maximum_boxes >= 0 && maximum_depth >= 0 && maximum_evaluations >= 0 ||
            throw(ArgumentError("box, depth and evaluation limits must be nonnegative"))
        maximum_records >= 1 || throw(ArgumentError("maximum_records must be positive"))
        isfinite(time_limit) && time_limit >= 0 ||
            throw(ArgumentError("time_limit must be finite and nonnegative"))
        maximum_precision >= 53 || throw(ArgumentError("maximum_precision must be at least 53"))
        new(maximum_boxes, maximum_depth, maximum_evaluations, maximum_records,
            Float64(time_limit), maximum_precision)
    end
end

"""An exact binary rectangle. BigFloat endpoints retain their stored precision."""
struct CertificationBox
    lower::NTuple{2,BigFloat}
    upper::NTuple{2,BigFloat}
    function CertificationBox(lower, upper)
        length(lower) == length(upper) == 2 || throw(ArgumentError("a box needs two coordinates"))
        all(i -> isfinite(lower[i]) && isfinite(upper[i]) && lower[i] <= upper[i], 1:2) ||
            throw(ArgumentError("box endpoints must be finite and ordered"))
        # Preserve BigFloat values verbatim; converting at ambient precision would
        # silently round a high-precision proof when it is serialized or replayed.
        exact(x::BigFloat) = deepcopy(x)
        exact(x) = BigFloat(x; precision=max(precision(BigFloat), 64))
        new(ntuple(i -> exact(lower[i]), 2), ntuple(i -> exact(upper[i]), 2))
    end
end

struct CertificationRoot
    witness::CertificationBox
    enclosure::CertificationBox
    precision::Int
    norm_bound::BigFloat
end

"""One replayable coverage obligation; IDs are positions in `nodes`."""
mutable struct CertificationNode
    box::CertificationBox
    parent::Int
    children::NTuple{2,Int}
    depth::Int
    precision::Int
    kind::Symbol
    root_id::Int
    reason::Symbol
end

"""
A fixed-parameter equilibrium proof on `[0,1/2]^2`.

Only `status == :certified` supplies an exact `equilibrium_count`. An unresolved
certificate preserves every pending root obligation in its proof tree. Roots
already proved to exist remain valid even when coverage is incomplete.
"""
struct EquilibriumCertificate{M}
    model::M
    options::CertificationOptions
    domain::CertificationBox
    roots::Vector{CertificationRoot}
    nodes::Vector{CertificationNode}
    status::Symbol
    equilibrium_count::Union{Nothing,Int}
    processed_boxes::Int
    evaluations::Int
    elapsed_seconds::Float64
    provenance::Dict{String,String}
end

struct _CertificationFailure <: Exception
    reason::Symbol
end

function _certification_rounding_check()
    _CIA.default_rounding() isa _CIA.IntervalRounding{:correct} ||
        throw(ArgumentError("certification requires IntervalArithmetic correct rounding"))
    return nothing
end

function _certification_valid(values)
    for x in values
        x isa _CIA.Interval && _CIA.isguaranteed(x) &&
            _CIA.decoration(x) == _CIA.com &&
            isfinite(_CIA.inf(x)) && isfinite(_CIA.sup(x)) &&
            _CIA.inf(x) <= _CIA.sup(x) ||
            throw(_CertificationFailure(:invalid_interval))
    end
    return values
end

_certification_type(x) = typeof(_CIA.inf(x))
_certification_constant(x, value) = _CIA.interval(_certification_type(x), value)
_certification_intervals(box::CertificationBox, ::Type{T}) where {T} =
    [_CIA.interval(T, box.lower[i], box.upper[i]) for i in 1:2]
_certification_box(x) = CertificationBox(ntuple(i -> _CIA.inf(x[i]), 2),
    ntuple(i -> _CIA.sup(x[i]), 2))
_certification_subset(a::CertificationBox, b::CertificationBox) =
    all(i -> b.lower[i] <= a.lower[i] <= a.upper[i] <= b.upper[i], 1:2)
_certification_disjoint(a::CertificationBox, b::CertificationBox) =
    any(i -> a.upper[i] < b.lower[i] || b.upper[i] < a.lower[i], 1:2)
_certification_equal(a::CertificationBox, b::CertificationBox) =
    a.lower == b.lower && a.upper == b.upper

function _certification_intersection(a::CertificationBox, b::CertificationBox)
    _certification_disjoint(a, b) && return nothing
    return CertificationBox(ntuple(i -> max(a.lower[i], b.lower[i]), 2),
        ntuple(i -> min(a.upper[i], b.upper[i]), 2))
end

# Stable monotone logistic image. Avoid exp overflow even on very wide boxes.
function _certification_logistic(z)
    _certification_valid((z,))
    T = _certification_type(z)
    one_i = _CIA.interval(T, 1)
    endpoint(t) = t >= 0 ? inv(one_i + exp(-_CIA.interval(T, t))) :
        (let q = exp(_CIA.interval(T, t)); q / (one_i + q) end)
    lo, hi = endpoint(_CIA.inf(z)), endpoint(_CIA.sup(z))
    _certification_valid((lo, hi))
    return _CIA.interval(T, _CIA.inf(lo), _CIA.sup(hi))
end

function _certification_response(r::LogisticResponse, u)
    a = _certification_constant(u, r.slope)
    z = a * (u - _certification_constant(u, r.threshold))
    value = _certification_logistic(z)
    derivative = a * value * _certification_logistic(-z)
    _certification_valid((value, derivative))
    return value, derivative
end

function _certification_response(r::FailureOfInhibitionResponse, u)
    a = _certification_constant(u, r.slope)
    onset = a * (u - _certification_constant(u, r.onset_threshold))
    failure = a * (u - _certification_constant(u, r.failure_threshold))
    separation = a * (_certification_constant(u, r.failure_threshold) -
        _certification_constant(u, r.onset_threshold))
    # This product is algebraically the logistic difference, with no subtraction
    # of two saturated near-one responses.
    value = _certification_logistic(onset) * _certification_logistic(-failure) *
        (-expm1(-separation))
    derivative = a * value *
        (_certification_logistic(-onset) - _certification_logistic(failure))
    _certification_valid((value, derivative))
    return value, derivative
end

function _certification_drive(model)
    drive = model.drive
    drive isa NoDrive && return (0, 0)
    drive isa PiecewiseConstantDrive && isempty(drive.pulses) && return drive.baseline
    throw(ArgumentError("certification requires NoDrive or a tonic drive without pulses"))
end

# Constant enclosures belong to this fixed model and precision. The cache is
# explicitly owned by the callback pair, never shared between certifications.
function _certification_callbacks(model)
    coupling, B = model.coupling, _certification_drive(model)
    e, i = model.excitatory.response, model.inhibitory.response
    ti = i isa LogisticResponse ? i.threshold : i.onset_threshold
    tf = i isa LogisticResponse ? i.threshold : i.failure_threshold
    raw = (coupling.e_to_e, coupling.i_to_e, coupling.e_to_i, coupling.i_to_i,
        B..., e.slope, e.threshold, i.slope, ti, tf)
    low = [_CIA.interval(Float64, v) for v in raw]
    high = Dict{Int,Vector{_CIA.Interval{BigFloat}}}()
    constants(::Type{Float64}) = low
    function constants(::Type{BigFloat})
        get!(high, precision(BigFloat)) do
            [_CIA.interval(BigFloat, v) for v in raw]
        end
    end
    function rate(r::LogisticResponse, u, a, theta, theta_off)
        _certification_logistic(a * (u - theta))
    end
    function rate(r::FailureOfInhibitionResponse, u, a, theta, theta_off)
        _certification_logistic(a * (u - theta)) *
            _certification_logistic(-a * (u - theta_off)) *
            (-expm1(-a * (theta_off - theta)))
    end
    function slope(r::LogisticResponse, u, a, theta, theta_off, value)
        a * value * _certification_logistic(-a * (u - theta))
    end
    function slope(r::FailureOfInhibitionResponse, u, a, theta, theta_off, value)
        a * value * (_certification_logistic(-a * (u - theta)) -
            _certification_logistic(a * (u - theta_off)))
    end
    function evaluate(box, want_jacobian)
        _certification_valid(box)
        E, I = box
        c = constants(_certification_type(E))
        u_E = c[1] * E - c[2] * I + c[5]
        u_I = c[3] * E - c[4] * I + c[6]
        re = rate(e, u_E, c[7], c[8], c[8])
        ri = rate(i, u_I, c[9], c[10], c[11])
        _certification_valid((re, ri))
        unit = _CIA.interval(_certification_type(E), 1)
        if !want_jacobian
            return _certification_valid([-E + (unit - E) * re, -I + (unit - I) * ri])
        end
        de = slope(e, u_E, c[7], c[8], c[8], re)
        di = slope(i, u_I, c[9], c[10], c[11], ri)
        return _certification_valid([
            -unit - re + (unit - E) * de * c[1]  -(unit - E) * de * c[2];
            (unit - I) * di * c[3]  -unit - ri - (unit - I) * di * c[4]])
    end
    return box -> evaluate(box, false), box -> evaluate(box, true)
end

function _certification_saturation(model, box)
    c(v) = _certification_constant(box[1], v)
    x = [_CIA.interval(_certification_type(box[1]), _CIA.mid(v)) for v in box]
    coupling, B = model.coupling, _certification_drive(model)
    u = (c(coupling.e_to_e) * x[1] - c(coupling.i_to_e) * x[2] + c(B[1]),
        c(coupling.e_to_i) * x[1] - c(coupling.i_to_i) * x[2] + c(B[2]))
    for (r, input) in zip((model.excitatory.response, model.inhibitory.response), u)
        v = first(_certification_response(r, input))
        (_CIA.inf(v) <= 0 || _CIA.sup(v) >= 1) && return true
    end
    return false
end

function _certification_krawczyk(field, jacobian, X)
    _certification_valid(X)
    T = _certification_type(X[1])
    point = [_CIA.interval(T, _CIA.mid(x)) for x in X]
    f, J = field(point), jacobian(X)
    length(f) == 2 && size(J) == (2, 2) || throw(_CertificationFailure(:invalid_shape))
    _certification_valid(f)
    _certification_valid(J)
    a, b, c, d = _CIA.mid(J[1, 1]), _CIA.mid(J[1, 2]),
        _CIA.mid(J[2, 1]), _CIA.mid(J[2, 2])
    det = a * d - b * c
    isfinite(det) && det != 0 || return nothing
    C = [_CIA.interval(T, d / det) _CIA.interval(T, -b / det);
         _CIA.interval(T, -c / det) _CIA.interval(T, a / det)]
    _certification_valid(C)
    # The rounded point preconditioner must itself be nonsingular.
    determinant = C[1, 1] * C[2, 2] - C[1, 2] * C[2, 1]
    _certification_valid((determinant,))
    _CIA.inf(determinant) <= 0 <= _CIA.sup(determinant) && return nothing
    unit, nil = _CIA.interval(T, 1), _CIA.interval(T, 0)
    R = [(i == j ? unit : nil) - (C[i, 1] * J[1, j] + C[i, 2] * J[2, j])
        for i in 1:2, j in 1:2]
    K = [point[i] - (C[i, 1] * f[1] + C[i, 2] * f[2]) +
        (R[i, 1] * (X[1] - point[1]) + R[i, 2] * (X[2] - point[2])) for i in 1:2]
    # Row sums use interval addition too: an ordinary floating sum could round
    # downward and falsely pass the strict contraction threshold.
    norms = [abs(R[i, 1]) + abs(R[i, 2]) for i in 1:2]
    _certification_valid(R)
    _certification_valid(K)
    _certification_valid(norms)
    return (box=_certification_box(K), norm_bound=maximum(_CIA.sup.(norms)))
end

function _certification_split(box, ::Type{T}) where {T}
    X = _certification_intervals(box, T)
    widths = [box.upper[i] - box.lower[i] for i in 1:2]
    axis = widths[1] >= widths[2] ? 1 : 2
    divider = _CIA.mid(X[axis])
    box.lower[axis] < divider < box.upper[axis] || return nothing
    a = CertificationBox(box.lower, ntuple(i -> i == axis ? divider : box.upper[i], 2))
    b = CertificationBox(ntuple(i -> i == axis ? divider : box.lower[i], 2), box.upper)
    return a, b
end

function _certification_widen(box, domain, ::Type{T}) where {T}
    X = _certification_intervals(box, T)
    two = _CIA.interval(T, 2)
    expanded = [X[i] + (X[i] - X[i]) / two for i in 1:2]
    _certification_valid(expanded)
    return _certification_intersection(_certification_box(expanded), domain)
end

function _certification_register!(roots, witness, test, precision; maximum_roots=typemax(Int))
    test.norm_bound < 1 && _certification_subset(test.box, witness) || return 0
    for (id, root) in pairs(roots)
        if _certification_subset(test.box, root.witness) ||
            _certification_subset(root.enclosure, witness)
            return id
        end
        _certification_disjoint(test.box, root.enclosure) || return 0
    end
    length(roots) < maximum_roots || throw(_CertificationFailure(:record_limit))
    push!(roots, CertificationRoot(witness, test.box, precision, BigFloat(test.norm_bound)))
    return length(roots)
end

_certification_next_precision(p, options) = p >= options.maximum_precision ? p :
    min(options.maximum_precision, p == 53 ? 128 : 2p)

function _certify_equilibria(field, jacobian;
    domain=CertificationBox((0, 0), (1//2, 1//2)), options=CertificationOptions(),
    model=nothing, precision_hint=box -> false,
    use_range_exclusion=true, use_contraction=true)
    _certification_rounding_check()
    nodes = [CertificationNode(domain, 0, (0, 0), 0, 53, :pending, 0, :none)]
    roots = CertificationRoot[]
    stack = [1]
    evaluations = Ref(0)
    started = time_ns()
    elapsed() = Float64(time_ns() - started) / 1e9
    function evaluate(callback, box)
        elapsed() < options.time_limit || throw(_CertificationFailure(:time_limit))
        evaluations[] < options.maximum_evaluations || throw(_CertificationFailure(:evaluation_limit))
        evaluations[] += 1
        _certification_rounding_check()
        result = callback(box)
        _certification_rounding_check()
        elapsed() < options.time_limit || throw(_CertificationFailure(:time_limit))
        return result
    end
    f(box) = evaluate(field, box)
    j(box) = evaluate(jacobian, box)
    processed = 0
    function addchild!(parent, box, precision, depth)
        push!(nodes, CertificationNode(box, parent, (0, 0), depth, precision, :pending, 0, :none))
        push!(stack, length(nodes))
        return length(nodes)
    end
    while !isempty(stack)
        id = pop!(stack)
        node = nodes[id]
        limit = processed >= options.maximum_boxes ? :box_limit :
            elapsed() >= options.time_limit ? :time_limit : :none
        if limit != :none
            node.kind, node.reason = :unresolved, limit
            continue
        end
        processed += 1
        retry = true
        while retry
            retry = false
            if elapsed() >= options.time_limit
                node.kind, node.reason = :unresolved, :time_limit
                break
            end
            p = node.precision
            T = p == 53 ? Float64 : BigFloat
            try
                setprecision(BigFloat, p) do
                    X = _certification_intervals(node.box, T)
                    values = f(X)
                    length(values) == 2 || throw(_CertificationFailure(:invalid_shape))
                    _certification_valid(values)
                    if use_range_exclusion && any(v -> !(_CIA.inf(v) <= 0 <= _CIA.sup(v)), values)
                        node.kind = :range_exclusion
                        return
                    end
                    if evaluate(precision_hint, X)
                        throw(_CertificationFailure(:saturation_precision_limit))
                    end
                    assigned = findfirst(root -> _certification_subset(node.box, root.witness), roots)
                    if assigned !== nothing
                        node.kind, node.root_id = :root_assignment, assigned
                        return
                    end
                    test = _certification_krawczyk(f, j, X)
                    if test !== nothing
                        if _certification_disjoint(test.box, node.box)
                            node.kind = :krawczyk_exclusion
                            return
                        end
                        contracted = _certification_intersection(test.box, node.box)
                        assigned = findfirst(root -> _certification_subset(contracted, root.witness), roots)
                        if assigned === nothing
                            root_id = _certification_register!(roots, node.box, test, p;
                                maximum_roots=options.maximum_records - length(nodes))
                            if root_id == 0
                                # An overlapping witness resolves roots exactly on an
                                # internal split plane without counting the root twice.
                                witness = _certification_widen(node.box, domain, T)
                                if !_certification_equal(witness, node.box)
                                    wt = _certification_krawczyk(f, j, _certification_intervals(witness, T))
                                    wt !== nothing && (root_id = _certification_register!(roots, witness, wt, p;
                                        maximum_roots=options.maximum_records - length(nodes)))
                                end
                            end
                            root_id > 0 && _certification_subset(contracted, roots[root_id].witness) && (assigned = root_id)
                        end
                        if assigned !== nothing
                            node.kind, node.root_id = :root_assignment, assigned
                            return
                        end
                        # Only contract if the new rectangle is materially smaller.
                        oldwidth = maximum(node.box.upper[i] - node.box.lower[i] for i in 1:2)
                        newwidth = maximum(contracted.upper[i] - contracted.lower[i] for i in 1:2)
                        if use_contraction && newwidth < oldwidth / 2
                            length(nodes) + length(roots) < options.maximum_records || throw(_CertificationFailure(:record_limit))
                            child = addchild!(id, contracted, p, node.depth)
                            node.kind, node.children = :contract, (child, 0)
                            return
                        end
                    end
                    if node.depth >= options.maximum_depth
                        throw(_CertificationFailure(:depth_limit))
                    end
                    split = _certification_split(node.box, T)
                    split === nothing && throw(_CertificationFailure(:precision_limit))
                    length(nodes) + length(roots) + 2 <= options.maximum_records || throw(_CertificationFailure(:record_limit))
                    left = addchild!(id, split[1], p, node.depth + 1)
                    right = addchild!(id, split[2], p, node.depth + 1)
                    node.kind, node.children = :split, (left, right)
                end
            catch err
                err isa _CertificationFailure || rethrow()
                next = _certification_next_precision(p, options)
                if err.reason in (:invalid_interval, :saturation_precision_limit, :precision_limit, :depth_limit) && next > p
                    node.precision = next
                    retry = true
                else
                    node.kind, node.reason = :unresolved, err.reason
                end
            end
        end
        if elapsed() >= options.time_limit && node.children == (0, 0)
            node.kind, node.reason, node.root_id = :unresolved, :time_limit, 0
        end
    end
    complete = all(n -> n.kind != :unresolved, nodes)
    provenance = isdefined(@__MODULE__, :_certificate_provenance) ? _certificate_provenance(model) : Dict{String,String}()
    return EquilibriumCertificate(model, options, domain, roots, nodes,
        complete ? :certified : :unresolved, complete ? length(roots) : nothing,
        processed, evaluations[], elapsed(), provenance)
end

"""
    certify_equilibria(model; options=CertificationOptions())

Certify all roots of the dimensionless balance equations on `[0,1/2]^2`,
treating stored finite parameters as exact numbers. Only constant tonic drives
are supported. This proves fixed points, not attractors or biological identities.
Float64 intervals are tried first; difficult local boxes use bounded BigFloat
precision. Every exhausted resource leaves explicit unresolved coverage.
"""
function certify_equilibria(model::PointModelParameters; options=CertificationOptions())
    _certification_drive(model)
    values = (model.excitatory.timescale, model.inhibitory.timescale,
        _model_numeric_values(model)..., _certification_drive(model)...)
    all(v -> !(v isa Bool) && v isa Union{Integer,Float32,Float64,BigFloat}, values) ||
        throw(ArgumentError("certification supports stored integer, Float32, Float64 and BigFloat parameters"))
    field, jacobian = _certification_callbacks(model)
    return _certify_equilibria(field, jacobian; options, model,
        precision_hint=box -> _certification_saturation(model, box))
end

function _certification_split_covers(parent, left, right)
    _certification_subset(left, parent) && _certification_subset(right, parent) || return false
    for axis in 1:2
        other = 3 - axis
        if left.lower[axis] == parent.lower[axis] && right.upper[axis] == parent.upper[axis] &&
            left.upper[axis] == right.lower[axis] &&
            left.lower[other] == right.lower[other] == parent.lower[other] &&
            left.upper[other] == right.upper[other] == parent.upper[other]
            return true
        end
    end
    return false
end

"""Replay all analytical predicates and the complete root-obligation tree."""
function _verify_certificate(field, jacobian, cert::EquilibriumCertificate)
    try
        _certification_rounding_check()
        if cert.model isa PointModelParameters
            _certification_equal(cert.domain, CertificationBox((0, 0), (1//2, 1//2))) || return false
        end
        isempty(cert.nodes) && return false
        _certification_equal(cert.nodes[1].box, cert.domain) || return false
        cert.nodes[1].parent == 0 || return false
        for (i, root) in pairs(cert.roots)
            _certification_subset(root.witness, cert.domain) || return false
            53 <= root.precision <= cert.options.maximum_precision || return false
            okay = setprecision(BigFloat, root.precision) do
                T = root.precision == 53 ? Float64 : BigFloat
                test = _certification_krawczyk(field, jacobian, _certification_intervals(root.witness, T))
                test !== nothing && test.norm_bound < 1 &&
                    _certification_subset(test.box, root.witness) &&
                    _certification_equal(test.box, root.enclosure) &&
                    BigFloat(test.norm_bound) == root.norm_bound
            end
            okay || return false
            all(j -> _certification_disjoint(root.enclosure, cert.roots[j].enclosure), 1:i-1) || return false
        end
        visited = falses(length(cert.nodes))
        stack = [1]
        while !isempty(stack)
            id = pop!(stack)
            1 <= id <= length(cert.nodes) && !visited[id] || return false
            visited[id] = true
            node = cert.nodes[id]
            53 <= node.precision <= cert.options.maximum_precision || return false
            okay = setprecision(BigFloat, node.precision) do
                T = node.precision == 53 ? Float64 : BigFloat
                X = _certification_intervals(node.box, T)
                children = filter(!iszero, collect(node.children))
                if node.kind in (:split, :contract)
                    length(children) == (node.kind == :split ? 2 : 1) || return false
                    all(c -> id < c <= length(cert.nodes) && cert.nodes[c].parent == id, children) || return false
                    if node.kind == :split
                        _certification_split_covers(node.box, cert.nodes[children[1]].box,
                            cert.nodes[children[2]].box) || return false
                    else
                        test = _certification_krawczyk(field, jacobian, X)
                        test === nothing && return false
                        intersection = _certification_intersection(test.box, node.box)
                        intersection !== nothing && _certification_subset(intersection, cert.nodes[children[1]].box) &&
                            _certification_subset(cert.nodes[children[1]].box, node.box) || return false
                    end
                    append!(stack, children)
                    return true
                end
                isempty(children) || return false
                if node.kind == :range_exclusion
                    values = field(X)
                    length(values) == 2 || return false
                    _certification_valid(values)
                    return any(v -> !(_CIA.inf(v) <= 0 <= _CIA.sup(v)), values)
                elseif node.kind == :krawczyk_exclusion
                    test = _certification_krawczyk(field, jacobian, X)
                    return test !== nothing && _certification_disjoint(test.box, node.box)
                elseif node.kind == :root_assignment
                    1 <= node.root_id <= length(cert.roots) || return false
                    witness = cert.roots[node.root_id].witness
                    _certification_subset(node.box, witness) && return true
                    test = _certification_krawczyk(field, jacobian, X)
                    test === nothing && return false
                    intersection = _certification_intersection(test.box, node.box)
                    return intersection !== nothing && _certification_subset(intersection, witness)
                elseif node.kind == :unresolved
                    return node.reason in (:box_limit, :time_limit, :evaluation_limit, :record_limit,
                        :invalid_interval, :invalid_shape, :saturation_precision_limit, :depth_limit, :precision_limit)
                end
                return false
            end
            okay || return false
        end
        all(visited) || return false
        complete = all(n -> n.kind != :unresolved, cert.nodes)
        return cert.status == (complete ? :certified : :unresolved) &&
            cert.equilibrium_count == (complete ? length(cert.roots) : nothing)
    catch
        return false
    end
end
