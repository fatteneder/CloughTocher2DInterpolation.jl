module CloughTocher2DInterpolation


using MiniQhull
using LinearAlgebra

export CloughTocher2DInterpolator



const NDIM = 2



function __precompie__()

    coordinates = [0,0,
                   0,1,
                   1,0,
                   1,1,
                   2,0,
                   2,1]
    values = Float64[1,2,3,4,5,6]
    I = CloughTocher2DInterpolator(coordinates, values)
    icoords = [0.5, 0.5,
               1.5, 0.5]
    I(icoords)

end





const libqhull_r   = MiniQhull.QhullMiniWrapper_jll.Qhull_jll.libqhull_r
# Copy of MiniQhull._delaunay where we also extract the variables
# last_newhigh, last_high, last_low, SCALElast from the qhull_handler
function my_delaunay(dim::Int32, numpoints::Int32, points::Array{Float64},
                     flags::Union{Nothing,AbstractString})
    numcells = Ref{Int32}()
    qh = MiniQhull.new_qhull_handler()
    qh == C_NULL && error("Qhull handler is null")
    cflags = flags===nothing ? C_NULL : flags
    ierror = MiniQhull.delaunay_init_and_compute(qh, dim, numpoints, points, numcells, cflags)
    ierror != 0 && error("Failure on delaunay_init_and_compute function: $(ierror)")
    cells = Matrix{Int32}(undef,dim+1,numcells[])
    ierror = MiniQhull.delaunay_fill_cells(qh, dim, numcells[], cells)
    ierror != 0 && error("Failure on delaunay_fill_cells function")
    last_newhigh = ccall((:qh_get_last_newhigh,libqhull_r), Cdouble, (Ptr{Cvoid},), qh)
    last_high    = ccall((:qh_get_last_high,libqhull_r), Cdouble, (Ptr{Cvoid},), qh)
    last_low     = ccall((:qh_get_last_low,libqhull_r), Cdouble, (Ptr{Cvoid},), qh)
    SCALElast    = ccall((:qh_get_SCALElast,libqhull_r), Cuint, (Ptr{Cvoid},), qh)
    ierror = MiniQhull.delaunay_free(qh)
    ierror != 0 && error("Failure on delaunay_free function")
    cells, last_newhigh, last_high, last_low, SCALElast
end





# copy of DelaunayInfo_t from scipy/scipy/spatial/_qhull.pxd
# with only the fields necessary for the interpolation algo
struct DelaunayInfo

    # inputs
    npoints::Int32
    points::Matrix{Float64}

    # from qhull
    nsimplex::Int32
    simplices::Matrix{Int32}
    # equations
    paraboloid_scale::Float64
    paraboloid_shift::Float64
    max_bound::Vector{Float64}
    min_bound::Vector{Float64}

    # manually computed
    neighbors::Matrix{Int32}
    transform::Array{Float64,3}
    vertex_neighbors_indices::Vector{Int32}
    vertex_neighbors_indptr::Vector{Int32}

end


function DelaunayInfo(points::AbstractVector{<:Real})
    size_check(points)
    return DelaunayInfo(reshape(float.(points), 2, Int(length(points)/2)))
end
function DelaunayInfo(points::AbstractVector{<:Complex})
    size_check(points)
    return DelaunayInfo(reshape(complex.(points), 2, Int(length(points)/2)))
end


function DelaunayInfo(points::Matrix{Float64})

    dim, _npoints = size(points)
    npoints = Int32(_npoints)

    # these options are used by scipy when running the 6 point 2D MWE
    flags = "qhull d Qt Q12 Qz Qbb Qc"
    _points = Array{Float64}(points)
    simplices, last_newhigh, last_high, last_low, SCALElast =
        my_delaunay(Int32(NDIM), Int32(npoints), _points, flags)

    # scipy/scipy/spatial/_qhull:_Qhull::get_simplex_facet_array likes to exchange the
    # first two indices of simplces to maintain a counter clockwise order
    # we enforce the same ordering here
    # detect orientation by looking at the sign of the z-component of the cross product of
    # two vectors connecting simplices[2]->simplices[1] and simplices[3]->simplices[2]
    for simp in eachcol(simplices)
        v1x = points[1,simp[2]] - points[1,simp[1]]
        v2x = points[1,simp[3]] - points[1,simp[2]]
        v1y = points[2,simp[2]] - points[2,simp[1]]
        v2y = points[2,simp[3]] - points[2,simp[2]]
        z = v1x * v2y - v1y * v2x
        @assert !isapprox(z,0.0)
        if z < 0 # swap
            simp[1], simp[2] = simp[2], simp[1]
        end
    end

    nsimplex = Int32(size(simplices)[2])

    min_bound = vec(minimum(points, dims=2))
    max_bound = vec(maximum(points, dims=2))

    paraboloid_scale, paraboloid_shift = if SCALElast != 0
        scale = last_newhigh / (last_high - last_low)
        shift = - last_low * scale
        scale, shift
    else
        1.0, 0.0
    end

    # -1 indices if no neighbor
    neighbors = -ones(Int32, nsimplex, NDIM+1)
    # indices of neighboring simplices
    for isimplex = 1:nsimplex
        v1,v2,v3 = simplices[:,isimplex]
        for (i,simp) in enumerate(eachcol(simplices))
            i == isimplex && continue
            if v1 in simp && v2 in simp
                neighbors[isimplex,3] = i
            elseif v1 in simp && v3 in simp
                neighbors[isimplex,2] = i
            elseif v2 in simp && v3 in simp
                neighbors[isimplex,1] = i
            end
        end
    end

    # Literal translation of scipy/scipy/spatial/_qhull.pyx
    # TODO Combine this block with next one
    # TODO Convert this into a dict.
    indptr_indices = [ Int32[] for _ = 1:npoints ]
    for i = 1:nsimplex, j = 1:NDIM+1, k = 1:NDIM+1
        ki = simplices[k,i]
        ji = simplices[j,i]
        if ji != ki
            ki in indptr_indices[ji] && continue
            push!(indptr_indices[ji], ki)
        end
    end

    # Literal translation of scipy/scipy/spatial/setlist.pxd:tocsr
    N = length(indptr_indices)+1
    vertex_neighbors_indptr  = zeros(Int32, N)
    vertex_neighbors_indices = zeros(Int32, sum(length, indptr_indices))
    pos = 1
    for i = 1:N-1
        vertex_neighbors_indptr[i] = pos
        for j = 1:length(indptr_indices[i])
            vertex_neighbors_indices[pos] = indptr_indices[i][j]
            pos += 1
        end
    end
    vertex_neighbors_indptr[end] = pos

    transform = get_barycentric_transforms(npoints, points, nsimplex, simplices, eps(Float64))

    d = DelaunayInfo(npoints, points,
                     nsimplex, simplices, #=equations,=#
                     paraboloid_scale, paraboloid_shift, max_bound, min_bound,
                     neighbors, transform, vertex_neighbors_indices, vertex_neighbors_indptr)
    return d

end


function get_barycentric_transforms(npoints, _points, nsimplex, simplices, eps)
    """
    Compute barycentric affine coordinate transformations for given
    simplices.

    Returns
    -------
    Tinvs : array, shape (nsimplex, ndim+1, ndim)
        Barycentric transforms for each simplex.

        Tinvs[i,:ndim,:ndim] contains inverse of the matrix ``T``,
        and Tinvs[i,ndim,:] contains the vector ``r_n`` (see below).

    Notes
    -----
    Barycentric transform from ``x`` to ``c`` is defined by::

        T c = x - r_n

    where the ``r_1, ..., r_n`` are the vertices of the simplex.
    The matrix ``T`` is defined by the condition::

        T e_j = r_j - r_n

    where ``e_j`` is the unit axis vector, e.g, ``e_2 = [0,1,0,0,...]``
    This implies that ``T_ij = (r_j - r_n)_i``.

    For the barycentric transforms, we need to compute the inverse
    matrix ``T^-1`` and store the vectors ``r_n`` for each vertex.
    These are stacked into the `Tinvs` returned.

    """

    points = reshape(_points, (NDIM,npoints))

    T = zeros(Float64, NDIM, NDIM)
    # Can't use view(Tinvs, 1:dim, 1:dim) below, because that does not have
    # contiugous memory layout
    Tinvs = zeros(Float64, nsimplex, NDIM+1, NDIM)
    # instead we use a buffer
    Tinv = zeros(Float64, NDIM, NDIM)

    # points are layed out as npoints x ndims
    # simplices are layed out as nsimplices x ndims

    for isimplex = 1:nsimplex
        fill!(Tinv, 0.0)
        for i = 1:NDIM
            for j = 1:NDIM
                s_j = simplices[j,isimplex]
                s_n = simplices[end,isimplex]
                T[i,j] = points[i,s_j] - points[i,s_n]
            end
            Tinv[i,i] = 1
        end

        luT = lu!(T)
        ldiv!(luT, Tinv)
        Tinvs[isimplex,1:NDIM,:] .= Tinv
        Tinvs[isimplex,end,:] .= points[:,simplices[end,isimplex]]

        # don't need to deal with degenerate simplices, because lu!
        # would have raised a SingularException already

    end

    return Tinvs

end






function _estimate_gradients_2d_global(d::DelaunayInfo, data, maxiter, tol, y)

    T = eltype(data)
    Q = zeros(Float64, 2*2)
    s = zeros(T, 2)
    r = zeros(T, 2)

    fill!(y, zero(T))

    ### Comment from scipy
    # Main point:
    #
    #    Z = sum_T sum_{E in T} int_E |W''|^2 = min!
    #
    # where W'' is the second derivative of the Clough-Tocher
    # interpolant to the direction of the edge E in triangle T.
    #
    # The minimization is done iteratively: for each vertex V,
    # the sum
    #
    #    Z_V = sum_{E connected to V} int_E |W''|^2
    #
    # is minimized separately, using existing values at other V.
    #
    # Since the interpolant can be written as
    #
    #     W(x) = f(x) + w(x)^T y
    #
    # where y = [ F_x(V); F_y(V) ], it is clear that the solution to
    # the local problem is is given as a solution of the 2x2 matrix
    # equation.
    #
    # Here, we use the Clough-Tocher interpolant, which restricted to
    # a single edge is
    #
    #     w(x) = (1 - x)**3   * f1
    #          + x*(1 - x)**2 * (df1 + 3*f1)
    #          + x**2*(1 - x) * (df2 + 3*f2)
    #          + x**3         * f2
    #
    # where f1, f2 are values at the vertices, and df1 and df2 are
    # derivatives along the edge (away from the vertices).
    #
    # As a consequence, one finds
    #
    #     L^3 int_{E} |W''|^2 = y^T A y + 2 B y + C
    #
    # with
    #
    #     A   = [4, -2; -2, 4]
    #     B   = [6*(f1 - f2), 6*(f2 - f1)]
    #     y   = [df1, df2]
    #     L   = length of edge E
    #
    # and C is not needed for minimization. Since df1 = dF1.E, df2 = -dF2.E,
    # with dF1 = [F_x(V_1), F_y(V_1)], and the edge vector E = V2 - V1,
    # we have
    #
    #     Z_V = dF1^T Q dF1 + 2 s.dF1 + const.
    #
    # which is minimized by
    #
    #     dF1 = -Q^{-1} s
    #
    # where
    #
    #     Q = sum_E [A_11 E E^T]/L_E^3 = 4 sum_E [E E^T]/L_E^3
    #     s = sum_E [ B_1 + A_21 df2] E /L_E^3
    #       = sum_E [ 6*(f1 - f2) + 2*(E.dF2)] E / L_E^3
    #

    # Gauss-Seidel
    for iiter in 1:maxiter
        err = 0
        for ipoint = 1:d.npoints
            fill!(Q, zero(T))
            fill!(s, zero(T))

            # walk over neighbours of given point
            for jpoint2 in d.vertex_neighbors_indptr[ipoint]:d.vertex_neighbors_indptr[ipoint+1]-1
                ipoint2 = d.vertex_neighbors_indices[jpoint2]

                # edge
                ex = d.points[2*ipoint2 - 1] - d.points[2*ipoint - 1]
                ey = d.points[2*ipoint2 + 0] - d.points[2*ipoint + 0]
                L = sqrt(ex^2 + ey^2)
                L3 = L*L*L

                # data at vertices
                f1 = data[ipoint]
                f2 = data[ipoint2]

                # scaled gradient projections on the edge
                df2 = -ex*y[2*ipoint2 - 1] - ey*y[2*ipoint2 + 0]

                # edge sum
                Q[1] += 4*ex*ex / L3
                Q[2] += 4*ex*ey / L3
                Q[4] += 4*ey*ey / L3

                s[1] += (6*(f1 - f2) - 2*df2) * ex / L3
                s[2] += (6*(f1 - f2) - 2*df2) * ey / L3
            end

            Q[3] = Q[2]

            # solve

            det = Q[1]*Q[4] - Q[2]*Q[3]
            r[1] = ( Q[4]*s[1] - Q[2]*s[2])/det
            r[2] = (-Q[3]*s[1] + Q[1]*s[2])/det

            change = max(abs(y[2*ipoint - 1] + r[1]),
                         abs(y[2*ipoint + 0] + r[2]))

            y[2*ipoint - 1] = -r[1]
            y[2*ipoint + 0] = -r[2]

            # relative/absolute error
            change /= max(1.0, max(abs(r[1]), abs(r[2])))
            err = max(err, change)
        end

        err < tol && return iiter + 1
    end

    # Didn't converge before maxiter
    return 0

end


function estimate_gradients_2d_global(info, y::AbstractArray{T}, maxiter=400, tol=1e-6) where T<:Complex
    rg = estimate_gradients_2d_global(info, real.(y), maxiter, tol)
    ig = estimate_gradients_2d_global(info, imag.(y), maxiter, tol)
    return rg + 1im * ig
end


function estimate_gradients_2d_global(info, y, maxiter=400, tol=1e-6)

    # length(y) != info.npoints && error("'y' has a wrong number of items")

    grad = zeros(eltype(y), 2, info.npoints)

    # scipy/scipy/interpolate/interpnd.pyx:estimate_gradients_2d_global
    # contains a useless loop 'for k in range(nvalues):` which only does one iteration
    # to call _estimate_gradients_2d_global
    ret = _estimate_gradients_2d_global(info, y, maxiter, tol, grad)
    if ret == 0
        @warn("Gradient estimation did not converge, the results may be inaccurate")
    end

    return grad

end






struct CloughTocher2DInterpolator{T<:Number}
    info::DelaunayInfo # triangulation
    values::Vector{T}
    grad::Matrix{T}
    fill_value::T
    # offset
    # scale
    function CloughTocher2DInterpolator(info, values, grad, fill_value)
        vT = eltype(values)
        gT = eltype(grad)
        fT = eltype(fill_value)
        T = promote_type(vT, gT, fT)
        if T <: Real
            return new{T}(info, float.(values), float.(grad), float(fill_value))
        elseif T <: Complex
            return new{T}(info, complex.(values), complex.(grad), complex(fill_value))
        else
            throw(ArgumentError("failed to normalize types of data fields, can't handle $T"))
        end
    end
end


function size_check(points::AbstractVector)
    if length(points) % NDIM != 0
        throw(ArgumentError("points must be of length 2*npoints"))
    end
end
function size_check(points::AbstractMatrix)
    if size(points)[1] != 2
        throw(ArgumentError("points must be a matrix of size (2,npoints), found $(size(points))"))
    end
end


function CloughTocher2DInterpolator(points::AbstractVector, values::AbstractVector; kwargs...)
    size_check(points)
    npoints = Int32(length(points)/2)
    return CloughTocher2DInterpolator(reshape(points,(2,npoints)), values; kwargs...)
end


function CloughTocher2DInterpolator(points::AbstractMatrix, values::AbstractVector{T};
        fill_value=NaN, tol=1e-6, maxiter=400, rescale=false) where T

    size_check(points)
    if size(points)[2] != length(values)
        throw(ArgumentError("mismatch between number of points and values, found $(size(points)[2]) vs. $(length(values))"))
    end

    # TODO Scipy provides a rescale arg. We need that?
    info = DelaunayInfo(float.(points))
    vals = T<:Real ? float.(values) : complex.(float.(real.(values)), float.(imag.(values)))
    grad = estimate_gradients_2d_global(info, vals, maxiter, tol)

    return CloughTocher2DInterpolator(info, values, grad, fill_value)
end







function (I::CloughTocher2DInterpolator{T})(intrp_points) where T
    size_check(intrp_points)
    intrp_values = similar(intrp_points, T, Int(length(intrp_points)/2))
    I(intrp_values, intrp_points) # inplace
    return intrp_values
end


# interpolate inplace of intrp_values
function (I::CloughTocher2DInterpolator{T})(intrp_values::AbstractVector{T}, _intrp_points) where T

    size_check(_intrp_points)
    n_intrp_points = Int(length(_intrp_points)/2)
    intrp_points = reshape(_intrp_points, 2, n_intrp_points)
    n_intrp_values = length(intrp_values)
    if n_intrp_values != n_intrp_points
        resize!(intrp_values, n_intrp_values)
    end

    buffer_c  = zeros(Float64, 3)
    buffer_f  = zeros(T, 3)
    buffer_df = zeros(T, 6)

    eeps = 100 * eps(Float64)
    eeps_broad = sqrt(eeps)

    for (i,pt) in enumerate(eachcol(intrp_points))

        # scipy uses qhull._find_simplex here, which we can't, because we can't
        # extract the hyperplane normals and offsets that define the facets
        # (these are grouped under self.equations in DelaunyInfo_t)
        isimplex = _find_simplex_bruteforce(I.info, buffer_c, pt, eeps, eeps_broad)

        # Clough-Tocher interpolation
        if isimplex == -1
            intrp_values[i] = I.fill_value
            continue
        end

        for j = 1:NDIM+1
            is               = I.info.simplices[j,isimplex]
            buffer_f[j]      = I.values[is]
            buffer_df[2*j-1] = I.grad[1,is]
            buffer_df[2*j+0] = I.grad[2,is]
        end
        w = _clough_tocher_2d_single(I.info, isimplex, buffer_c, buffer_f, buffer_df)
        intrp_values[i] = w

    end

end


function find_simplex(d::DelaunayInfo, pt::NTuple{2}, tol=nothing)
    eeps = isnothing(tol) ? 100 * eps(Float64) : tol
    eeps_broad = sqrt(eeps)
    buffer_c = zeros(Float64, 3)
    isimplex = _find_simplex_bruteforce(d, buffer_c, pt, eeps, eeps_broad)
    return isimplex
end


function find_simplex(d::DelaunayInfo, _points::AbstractArray, tol=nothing)

    size_check(_points)
    npoints = Int(length(_points)/2)
    points = reshape(_points, 2, npoints)

    eeps = isnothing(tol) ? 100 * eps(Float64) : tol
    eeps_broad = sqrt(eps)

    buffer_c = zeros(3)

    isimplices = [ _find_simplex_bruteforce(d, buffer_c, pt, eeps, eeps_broad)
                   for pt in points ]

    return isimplices
end




# function _find_simplex(d, c, x, start, eps, eps_broad)
#     """
#     Find simplex containing point `x` by walking the triangulation.
#
#     Notes
#     -----
#     This algorithm is similar as used by ``qh_findbest``.  The idea
#     is the following:
#
#     1. Delaunay triangulation is a projection of the lower half of a convex
#        hull, of points lifted on a paraboloid.
#
#        Simplices in the triangulation == facets on the convex hull.
#
#     2. If a point belongs to a given simplex in the triangulation,
#        its image on the paraboloid is on the positive side of
#        the corresponding facet.
#
#     3. However, it is not necessarily the *only* such facet.
#
#     4. Also, it is not necessarily the facet whose hyperplane distance
#        to the point on the paraboloid is the largest.
#
#     ..note::
#
#         If I'm not mistaken, `qh_findbestfacet` finds a facet for
#         which the plane distance is maximized -- so it doesn't always
#         return the simplex containing the point given. For example:
#
#         >>> p = np.array([(1 - 1e-4, 0.1)])
#         >>> points = np.array([(0,0), (1, 1), (1, 0), (0.99189033, 0.37674127),
#         ...                    (0.99440079, 0.45182168)], dtype=np.double)
#         >>> tri = qhull.delaunay(points)
#         >>> tri.simplices
#         array([[4, 1, 0],
#                [4, 2, 1],
#                [3, 2, 0],
#                [3, 4, 0],
#                [3, 4, 2]])
#         >>> dist = qhull.plane_distance(tri, p)
#         >>> dist
#         array([[-0.12231439,  0.00184863,  0.01049659, -0.04714842,
#                 0.00425905]])
#         >>> tri.simplices[dist.argmax()]
#         array([3, 2, 0]
#
#         Now, the maximally positive-distant simplex is [3, 2, 0], although
#         the simplex containing the point is [4, 2, 1].
#
#     In this algorithm, we walk around the tessellation trying to locate
#     a positive-distant facet. After finding one, we fall back to a
#     directed search.
#
#     """
#
#     _is_point_fully_outside(d, x, eps) && return -1, start
#     d.nsimplex <= 0 && return -1, start
#
#     isimplex = start
#
#     if isimplex < 0 || isimplex >= d.nsimplex
#         isimplex = 1
#     end
#
#     # Lift point to paraboloid
#     z = _lift_point(d, x)
#
#     # Walk the tessellation searching for a facet with a positive planar distance
#     best_dist = _distplane(d, isimplex, z)
#     changed = 1
#     while changed
#         best_dist > 0 && break
#         changed = 0
#         for k in 1:NDIM+1
#             # ineigh = d.neighbors[(NDIM+1)*isimplex + k]
#             ineigh = d.neighbors[isimplex,k]
#             ineigh == -1 && continue
#             dist = _distplane(d, ineigh, z)
#
#             # Note addition of eps -- otherwise, this code does not
#             # necessarily terminate! The compiler may use extended
#             # accuracy of the FPU so that (dist > best_dist), but
#             # after storing to double size, dist == best_dist,
#             # resulting to non-terminating loop
#
#             if dist > best_dist + eps*(1 + abs(best_dist))
#                 # Note: this is intentional: we jump in the middle of the cycle,
#                 #       and continue the cycle from the next k.
#                 #
#                 #       This apparently sweeps the different directions more
#                 #       efficiently. We don't need full accuracy, since we do
#                 #       a directed search afterwards in any case.
#                 isimplex = ineigh
#                 best_dist = dist
#                 changed = 1
#             end
#         end
#     end
#
#     # We should now be somewhere near the simplex containing the point,
#     # locate it with a directed search
#     return _find_simplex_directed(d, c, x, isimplex, eps, eps_broad), start
#
# end
#
#
# function _find_simplex_directed(d, c, x, start, eps, eps_broad)
#     """
#     Find simplex containing point `x` via a directed walk in the tessellation.
#
#     If the simplex is found, the array `c` is filled with the corresponding
#     barycentric coordinates.
#
#     Notes
#     -----
#
#     The idea here is the following:
#
#     1) In a simplex, the k-th neighbour is opposite the k-th vertex.
#        Call the ridge between them the k-th ridge.
#
#     2) If the k-th barycentric coordinate of the target point is negative,
#        then the k-th vertex and the target point lie on the opposite sides
#        of the k-th ridge.
#
#     3) Consequently, the k-th neighbour simplex is *closer* to the target point
#        than the present simplex, if projected on the normal of the k-th ridge.
#
#     4) In a regular tessellation, hopping to any such direction is OK.
#
#        Also, if one of the negative-coordinate neighbors happens to be -1,
#        then the target point is outside the tessellation (because the
#        tessellation is convex!).
#
#     5) If all barycentric coordinates are in [-eps, 1+eps], we have found the
#        simplex containing the target point.
#
#     6) If all barycentric coordinates are non-negative but 5) is not true,
#        we are in an inconsistent situation -- this should never happen.
#
#     This may however enter an infinite loop due to rounding errors in
#     the computation of the barycentric coordinates, so the iteration
#     count needs to be limited, and a fallback to brute force provided.
#
#     """
#
#     isimplex = start
#
#     if isimplex < 0 || isimplex >= d.nsimplex
#         isimplex = 1
#     end
#
#     # The maximum iteration count: it should be large enough so that
#     # the algorithm usually succeeds, but smaller than nsimplex so
#     # that for the cases where the algorithm fails, the main cost
#     # still comes from the brute force search.
#
#     for cycle_k = 1:1+Int(d.nsimplex/4)
#         if isimplex == -1
#             break
#         end
#
#         # transform = d.transform + isimplex*ndim*(ndim+1)
#
#         inside = 1
#         # for k in range(ndim+1)
#         for k = 1:ndim+1
#             _barycentric_coordinate_single(ndim, transform, x, c, k)
#
#             if c[k] < -eps
#                 # The target point is in the direction of neighbor `k`!
#                 # m = d.neighbors[(ndim+1)*isimplex + k]
#                 m = d.neighbors[isimplex,k]
#                 if m == -1
#                     # The point is outside the triangulation: bail out
#                     # start[0] = isimplex
#                     return isimplex, start
#                 end
#
#                 isimplex = m
#                 inside = -1
#                 break
#             elseif c[k] <= 1 + eps
#                 # we're inside this simplex
#             else
#                 # we're outside (or the coordinate is nan; a degenerate simplex)
#                 inside = 0
#             end
#         end
#
#         if inside == -1
#             # hopped to another simplex
#             continue
#         elseif inside == 1
#             # we've found the right one!
#             break
#         else
#             # we've failed utterly (degenerate simplices in the way).
#             # fall back to brute force
#             error()
#             isimplex = _find_simplex_bruteforce(d, c, x, eps, eps_broad)
#             break
#         end
#     end
#     # Not sure why there is an extra else: block in the scipy version, because
#     # the previous block is a for loop ???
#     # else:
#     #     # the algorithm failed to converge -- fall back to brute force
#     #     isimplex = _find_simplex_bruteforce(d, c, x, eps, eps_broad)
#     # end
#
#     return isimplex, start
#
# end


function _find_simplex_bruteforce(d, c, x, eps, eps_broad)
    """
    Find simplex containing point `x` by going through all simplices.

    """

    if _is_point_fully_outside(d, x, eps)
        return -1
    end

    for isimplex = 1:d.nsimplex
        transform = view(d.transform,isimplex,:,:)

        if !any(isnan, transform)
            inside = _barycentric_inside(transform, x, c, eps)
            if inside
                return isimplex
            end
        else
            # transform is invalid (nan, implying degenerate simplex)

            # we replace this inside-check by a check of the neighbors
            # with a larger epsilon

            for k = 1:NDIM+1
                ineighbor = d.neighbors[isimplex,k]
                if ineighbor == -1
                    continue
                end

                transform = view(d.transform,ineighbor,:,:)
                # TODO Move the nan checks to the constructor
                if any(isnan, transform)
                    # another bad simplex
                    continue
                end

                _barycentric_coordinates(NDIM, transform, x, c)

                # Check that the point lies (almost) inside the
                # neigbor simplex
                inside = 1
                for m = 1:NDIM+1
                    if d.neighbors[ineighbor,m] == isimplex
                        # allow extra leeway towards isimplex
                        if !(-eps_broad <= c[m] <= 1 + eps)
                            inside = 0
                            break
                        end
                    else
                        # normal check
                        if !(-eps <= c[m] <= 1 + eps)
                            inside = 0
                            break
                        end
                    end
                end
                if inside == 1
                    return ineighbor
                end
            end
        end
    end

    return -1
end


function _barycentric_inside(transform, x, c, eps)
    """
    Check whether point is inside a simplex, using barycentric
    coordinates.  `c` will be filled with barycentric coordinates, if
    the point happens to be inside.

    """
    c[NDIM+1] = 1.0
    for i = 1:NDIM
        c[i] = 0
        for j = 1:NDIM
            c[i] += transform[i,j] * (x[j] - transform[NDIM+1,j])
        end
        c[NDIM+1] -= c[i]

        if !(-eps <= c[i] <= 1 + eps)
            return false
        end
    end
    if !(-eps <= c[NDIM+1] <= 1 + eps)
        return false
    end
    return true
end


#### only needed for _find_simplex_directed
# function _barycentric_coordinate_single(ndim, transform, x, c, i)
#     """
#     Compute a single barycentric coordinate.
#
#     Before the ndim+1'th coordinate can be computed, the other must have
#     been computed earlier.
#
#     """
#     if i == ndim
#         c[ndim] = 1.0
#         for j = 1:ndim
#             c[ndim] -= c[j]
#         end
#     else
#         c[i] = 0
#         for j = 1:ndim
#             # c[i] += transform[ndim*i + j] * (x[j] - transform[ndim*ndim + j])
#             c[i] += transform[i,j] * (x[j] - transform[ndim,j])
#         end
#     end
# end


function _barycentric_coordinates(transform, x, c)
    """
    Compute barycentric coordinates.

    """
    c[NDIM] = 1.0
    for i = 1:NDIM
        c[i] = 0
        for j in 1:NDIM
            c[i] += transform[j,i] * (x[j] - transform[j,NDIM])
        end
        c[NDIM] -= c[i]
    end
end


function _is_point_fully_outside(d, x, eps)
    """
    Is the point outside the bounding box of the triangulation?

    """
    (x[1] < d.min_bound[1] - eps || x[1] > d.max_bound[1] + eps) && return true
    (x[2] < d.min_bound[2] - eps || x[2] > d.max_bound[2] + eps) && return true
    return false
end


#### only needed for _find_simplex
# function _lift_point(d, x)
#     z = zeros(eltype(x), d.ndim)
#     # z[d.ndim] = 0
#     for i in 1:d.ndim
#         z[i] = x[i]
#         z[d.ndim] += x[i]^2
#     end
#     z[d.ndim] *= d.paraboloid_scale
#     z[d.ndim] += d.paraboloid_shift
#     return
# end
#
#
#### only needed for _find_simplex
# function _distplane(d, isimplex, point)
#     # TODO Fix indexing
#     dist = d.equations[isimplex*(d.ndim+2) + d.ndim+1]
#     for k in 1:ndim+1
#         dist += d.equations[isimplex*(d.ndim+2) + k] * point[k]
#     end
#     return dist
# end


function _clough_tocher_2d_single(d #=delaunay=#, isimplex, b, f, df)

    e12x = (+ d.points[1 + 2*(d.simplices[3*(isimplex-1) + 2]-1)]
            - d.points[1 + 2*(d.simplices[3*(isimplex-1) + 1]-1)])
    e12y = (+ d.points[2 + 2*(d.simplices[3*(isimplex-1) + 2]-1)]
            - d.points[2 + 2*(d.simplices[3*(isimplex-1) + 1]-1)])

    e23x = (+ d.points[1 + 2*(d.simplices[3*(isimplex-1) + 3]-1)]
            - d.points[1 + 2*(d.simplices[3*(isimplex-1) + 2]-1)])
    e23y = (+ d.points[2 + 2*(d.simplices[3*(isimplex-1) + 3]-1)]
            - d.points[2 + 2*(d.simplices[3*(isimplex-1) + 2]-1)])

    e31x = (+ d.points[1 + 2*(d.simplices[3*(isimplex-1) + 1]-1)]
            - d.points[1 + 2*(d.simplices[3*(isimplex-1) + 3]-1)])
    e31y = (+ d.points[2 + 2*(d.simplices[3*(isimplex-1) + 1]-1)]
            - d.points[2 + 2*(d.simplices[3*(isimplex-1) + 3]-1)])

    f1 = f[1]
    f2 = f[2]
    f3 = f[3]

    df12 = +(df[2*1-1]*e12x + df[2*1+0]*e12y)
    df21 = -(df[2*2-1]*e12x + df[2*2+0]*e12y)
    df23 = +(df[2*2-1]*e23x + df[2*2+0]*e23y)
    df32 = -(df[2*3-1]*e23x + df[2*3+0]*e23y)
    df31 = +(df[2*3-1]*e31x + df[2*3+0]*e31y)
    df13 = -(df[2*1-1]*e31x + df[2*1+0]*e31y)

    c3000 = f1
    c2100 = (df12 + 3*c3000)/3
    c2010 = (df13 + 3*c3000)/3
    c0300 = f2
    c1200 = (df21 + 3*c0300)/3
    c0210 = (df23 + 3*c0300)/3
    c0030 = f3
    c1020 = (df31 + 3*c0030)/3
    c0120 = (df32 + 3*c0030)/3

    c2001 = (c2100 + c2010 + c3000)/3
    c0201 = (c1200 + c0300 + c0210)/3
    c0021 = (c1020 + c0120 + c0030)/3

    #
    # Now, we need to impose the condition that the gradient of the spline
    # to some direction `w` is a linear function along the edge.
    #
    # As long as two neighbouring triangles agree on the choice of the
    # direction `w`, this ensures global C1 differentiability.
    # Otherwise, the choice of the direction is arbitrary (except that
    # it should not point along the edge, of course).
    #
    # In [CT]_, it is suggested to pick `w` as the normal of the edge.
    # This choice is given by the formulas
    #
    #    w_12 = E_24 + g[0] * E_23
    #    w_23 = E_34 + g[1] * E_31
    #    w_31 = E_14 + g[2] * E_12
    #
    #    g[0] = -(e24x*e23x + e24y*e23y) / (e23x**2 + e23y**2)
    #    g[1] = -(e34x*e31x + e34y*e31y) / (e31x**2 + e31y**2)
    #    g[2] = -(e14x*e12x + e14y*e12y) / (e12x**2 + e12y**2)
    #
    # However, this choice gives an interpolant that is *not*
    # invariant under affine transforms. This has some bad
    # consequences: for a very narrow triangle, the spline can
    # develops huge oscillations. For instance, with the input data
    #
    #     [(0, 0), (0, 1), (eps, eps)],   eps = 0.01
    #     F  = [0, 0, 1]
    #     dF = [(0,0), (0,0), (0,0)]
    #
    # one observes that as eps -> 0, the absolute maximum value of the
    # interpolant approaches infinity.
    #
    # So below, we aim to pick affine invariant `g[k]`.
    # We choose
    #
    #     w = V_4' - V_4
    #
    # where V_4 is the centroid of the current triangle, and V_4' the
    # centroid of the neighbour. Since this quantity transforms similarly
    # as the gradient under affine transforms, the resulting interpolant
    # is affine-invariant. Moreover, two neighbouring triangles clearly
    # always agree on the choice of `w` (sign is unimportant), and so
    # this choice also makes the interpolant C1.
    #
    # The drawback here is a performance penalty, since we need to
    # peek into neighbouring triangles.
    #

    y = zeros(Float64, 2)
    c = zeros(Float64, 3)
    g = zeros(Float64, 3)
    for k = 1:3
        itri = d.neighbors[3*(isimplex-1) + k]

        if itri == -1
            # No neighbour.
            # Compute derivative to the centroid direction (e_12 + e_13)/2.
            g[k] = -1.0/2.0
            continue
        end

        # Centroid of the neighbour, in our local barycentric coordinates

        y[1] = (+ d.points[1 + 2*(d.simplices[3*(itri-1) + 1]-1)]
                + d.points[1 + 2*(d.simplices[3*(itri-1) + 2]-1)]
                + d.points[1 + 2*(d.simplices[3*(itri-1) + 3]-1)]) / 3

        y[2] = (+ d.points[2 + 2*(d.simplices[3*(itri-1) + 1]-1)]
                + d.points[2 + 2*(d.simplices[3*(itri-1) + 2]-1)]
                + d.points[2 + 2*(d.simplices[3*(itri-1) + 3]-1)]) / 3

        transform = view(d.transform,isimplex,:,:)
        _barycentric_coordinates(transform, y, c)

        # Rewrite V_4'-V_4 = const*[(V_4-V_2) + g_i*(V_3 - V_2)]

        # Now, observe that the results can be written *in terms of
        # barycentric coordinates*. Barycentric coordinates stay
        # invariant under affine transformations, so we can directly
        # conclude that the choice below is affine-invariant.

        if k == 1
            g[k] = (2*c[3] + c[2] - 1) / (2 - 3*c[3] - 3*c[2])
        elseif k == 2
            g[k] = (2*c[1] + c[3] - 1) / (2 - 3*c[1] - 3*c[3])
        elseif k == 3
            g[k] = (2*c[2] + c[1] - 1) / (2 - 3*c[2] - 3*c[1])
        end

    end

    c0111 = (g[1]*(-c0300 + 3*c0210 - 3*c0120 + c0030)
             + (-c0300 + 2*c0210 - c0120 + c0021 + c0201))/2
    c1011 = (g[2]*(-c0030 + 3*c1020 - 3*c2010 + c3000)
             + (-c0030 + 2*c1020 - c2010 + c2001 + c0021))/2
    c1101 = (g[3]*(-c3000 + 3*c2100 - 3*c1200 + c0300)
             + (-c3000 + 2*c2100 - c1200 + c2001 + c0201))/2

    c1002 = (c1101 + c1011 + c2001)/3
    c0102 = (c1101 + c0111 + c0201)/3
    c0012 = (c1011 + c0111 + c0021)/3

    c0003 = (c1002 + c0102 + c0012)/3

    # extended barycentric coordinates
    minval = minimum(b)

    b1 = b[1] - minval
    b2 = b[2] - minval
    b3 = b[3] - minval
    b4 = 3*minval

    # evaluate the polynomial -- the stupid and ugly way to do it,
    # one of the 4 coordinates is in fact zero
    w = (b1^3*c3000 + 3*b1^2*b2*c2100 + 3*b1^2*b3*c2010 +
         3*b1^2*b4*c2001 + 3*b1*b2^2*c1200 +
         6*b1*b2*b4*c1101 + 3*b1*b3^2*c1020 + 6*b1*b3*b4*c1011 +
         3*b1*b4^2*c1002 + b2^3*c0300 + 3*b2^2*b3*c0210 +
         3*b2^2*b4*c0201 + 3*b2*b3^2*c0120 + 6*b2*b3*b4*c0111 +
         3*b2*b4^2*c0102 + b3^3*c0030 + 3*b3^2*b4*c0021 +
         3*b3*b4^2*c0012 + b4^3*c0003)

    return w

end


end # module CloughTocher2DInterpolation
