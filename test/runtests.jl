using Test
using Random
using CloughTocher2DInterpolation
import CloughTocher2DInterpolation: DelaunayInfo, find_simplex, NDIM


# Below test cases were taken from
# - scip/scipy/spatial/tests/test_qhull.py
# - scip/scipy/interpolate/tests/test_interpnd.py


### scipy/spatial/tests/test_qhull: TestUtilities::find_simplex
@testset "DelaunayInfo: find simplices" begin

    pts = [0,0,0,1,1,1,1,0]
    d = DelaunayInfo(pts)
    @test d.simplices == [2 4 3; 4 2 1]'

    for pt_i in [ (0.25, 0.25, 2),
                  (0.75, 0.75, 1),
                  (0.3,  0.2,  2) ]
        pt, i = pt_i[1:2], pt_i[3]
        isimplex = find_simplex(d, pt)
        @test isimplex == i
    end

end


### scipy/spatial/tests/test_qhull: TestVertexNeighborVertices::_check
function check_neighbors(d)
    expected = [ Set{Int32}() for _ = 1:d.npoints ]
    for simp in eachcol(d.simplices)
        for a in simp, b in simp
            if a != b
                push!(expected[a], b)
            end
        end
    end
    indptr, indices = d.vertex_neighbors_indptr, d.vertex_neighbors_indices
    got = [ Set(indices[indptr[i]:indptr[i+1]-1]) for i in 1:length(indptr)-1 ]
    @test got == expected
    return
end


@testset "Delaunay: neighbor vertices" begin

    ### scipy/spatial/tests/test_qhull: TestVertexNeighborVertices::test_triangle
    points = [0,0,0,1,1,0]
    d = DelaunayInfo(points)
    check_neighbors(d)

    ### scipy/spatial/tests/test_qhull: TestVertexNeighborVertices::test_rectangle
    points = [0,0,0,1,1,1,1,0]
    d = DelaunayInfo(points)
    check_neighbors(d)

    ### scipy/spatial/tests/test_qhull: TestVertexNeighborVertices::test_complicated
    points = [0,0,0,1,1,1,1,0,0.5,0.5,0.9,0.5]
    d = DelaunayInfo(points)
    check_neighbors(d)

end


@testset "DelaunayInfo: test triangulation" begin

    ### scipy/spatial/tests/test_qhull: TestDelaunay::test_nd_simplex
    # simple smoke test: triangulate a 2-dimensional simplex
    points = [1,0,0,1,1,1]
    d = DelaunayInfo(points)
    @test sort(d.simplices[:]) == Int32.(collect(1:3))
    @test d.neighbors[:] == -ones(Int32,3)

    ### scipy/spatial/tests/test_qhull: TestDelaunay::test_2d_square
    # simple smoke test: 2d square
    points = [0,0,0,1,1,1,1,0]
    d = DelaunayInfo(points)
    @test d.simplices == [2 4 3; 4 2 1]'
    @test d.neighbors == [-1 -1 2; -1 -1 1]

    ### scipy/spatial/tests/test_qhull: TestDelaunay::test_duplicate_points
    # shouldn't fail on duplicate points
    x = [0,1,0,1]
    y = [0,0,1,1]
    xp = vcat(x,x)
    yp = vcat(y,y)
    points = [ xy[i] for xy in zip(x,y) for i = 1:2 ]
    d = DelaunayInfo(points)
    points = [ xy[i] for xy in zip(xp,yp) for i = 1:2 ]
    d = DelaunayInfo(points)

    ### scipy/spatial/tests/test_qhull: TestDelaunay::test_pathological
    # both should succeed
    pathological_data_1 = reduce(vcat,[
        [-3.14,-3.14], [-3.14,-2.36], [-3.14,-1.57], [-3.14,-0.79],
        [-3.14,0.0], [-3.14,0.79], [-3.14,1.57], [-3.14,2.36],
        [-3.14,3.14], [-2.36,-3.14], [-2.36,-2.36], [-2.36,-1.57],
        [-2.36,-0.79], [-2.36,0.0], [-2.36,0.79], [-2.36,1.57],
        [-2.36,2.36], [-2.36,3.14], [-1.57,-0.79], [-1.57,0.79],
        [-1.57,-1.57], [-1.57,0.0], [-1.57,1.57], [-1.57,-3.14],
        [-1.57,-2.36], [-1.57,2.36], [-1.57,3.14], [-0.79,-1.57],
        [-0.79,1.57], [-0.79,-3.14], [-0.79,-2.36], [-0.79,-0.79],
        [-0.79,0.0], [-0.79,0.79], [-0.79,2.36], [-0.79,3.14],
        [0.0,-3.14], [0.0,-2.36], [0.0,-1.57], [0.0,-0.79], [0.0,0.0],
        [0.0,0.79], [0.0,1.57], [0.0,2.36], [0.0,3.14], [0.79,-3.14],
        [0.79,-2.36], [0.79,-0.79], [0.79,0.0], [0.79,0.79],
        [0.79,2.36], [0.79,3.14], [0.79,-1.57], [0.79,1.57],
        [1.57,-3.14], [1.57,-2.36], [1.57,2.36], [1.57,3.14],
        [1.57,-1.57], [1.57,0.0], [1.57,1.57], [1.57,-0.79],
        [1.57,0.79], [2.36,-3.14], [2.36,-2.36], [2.36,-1.57],
        [2.36,-0.79], [2.36,0.0], [2.36,0.79], [2.36,1.57],
        [2.36,2.36], [2.36,3.14], [3.14,-3.14], [3.14,-2.36],
        [3.14,-1.57], [3.14,-0.79], [3.14,0.0], [3.14,0.79],
        [3.14,1.57], [3.14,2.36], [3.14,3.14] ])
    d = DelaunayInfo(pathological_data_1)
    @test maximum(d.points[d.simplices]) ≈ maximum(pathological_data_1)
    @test minimum(d.points[d.simplices]) ≈ minimum(pathological_data_1)

    pathological_data_2 = reduce(vcat,[
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 0], [0, 1],
        [1, -1 - eps(Float64)], [1, 0], [1, 1] ])
    d = DelaunayInfo(pathological_data_2)
    @test maximum(d.points[d.simplices]) ≈ maximum(pathological_data_2)
    @test minimum(d.points[d.simplices]) ≈ minimum(pathological_data_2)

end


### scipy/interpolate/tests/test_interpnd: TestCloughTocher2DInterpolator::_check_accuracy
function check_accuracy(func; x=nothing, intrp_tol=1e-6, rescale=false, kwargs...)

    if isnothing(x)
        x = [0,0, 0,1, 1,0, 1,1, 0.25,0.75, 0.6,0.8, 0.5,0.2]
    end
    x = reshape(x, (2,Int(length(x)/2)))
    y = [ func(pt...) for pt in eachcol(x) ]

    ip = CloughTocher2DInterpolator(x, y, tol=intrp_tol, rescale=rescale)

    mt = Random.MersenneTwister(1234)
    p = rand(mt, 2, 50)

    a = ip(p)
    b = [ func(pt...) for pt in eachcol(p) ]

    @test all(isapprox.(a, b; kwargs...))

end


@testset "CloughTocher2DInterpolator" begin


    ### scipy/interpolate/tests/test_interpnd: TestCloughTocher2DInterpolator::test_linear_smoketest
    # Should be exact for linear functions, independent of triangulation
    funcs = [
        (x, y) -> 0*x + 1,
        (x, y) -> 0 + x,
        (x, y) -> -2 + y,
        (x, y) -> 3 + 3*x + 14.15*y,
    ]
    for (j, func) in enumerate(funcs)
        check_accuracy(func; intrp_tol=1e-13, atol=1e-7, rtol=1e-7)
        check_accuracy(func; intrp_tol=1e-13, atol=1e-7, rtol=1e-7, rescale=true)
    end

    # ### scipy/interpolate/tests/test_interpnd: TestCloughTocher2DInterpolator::test_quadratic_smoketest
    # # Should be reasonably accurate for quadratic functions
    funcs = [
        (x, y) -> x^2,
        (x, y) -> y^2,
        (x, y) -> x^2 - y^2,
        (x, y) -> x*y,
    ]
    for (j, func) in enumerate(funcs)
        # scipy can get away with atol=0.22 here ... could this be because we use a
        # different RNG here?
        check_accuracy(func; intrp_tol=1e-9, atol=0.29, rtol=0.0)
        check_accuracy(func; intrp_tol=1e-9, atol=0.29, rtol=0.0, rescale=true)
    end

    ### scipy/interpolate/tests/test_interpnd: TestCloughTocher2DInterpolator::test_tri_input
    # Test (complex) interpolation onto reference points
    x = [0,0,-0.5,-0.5,-0.5,0.5,0.5,0.5,0.25,0.3]
    y = collect(0:4)
    y = @. y - 3*im * y
    yi = CloughTocher2DInterpolator(x, y)(x)
    @test y ≈ yi
    yi = CloughTocher2DInterpolator(x, y, rescale=true)(x)
    @test y ≈ yi

    ### scipy/interpolate/tests/test_interpnd: TestCloughTocher2DInterpolator::test_dense
    # dense
    funcs = [
        (x, y) -> x^2,
        (x, y) -> y^2,
        (x, y) -> x^2 - y^2,
        (x, y) -> x*y,
        (x, y) -> cos(2*pi*x)*sin(2*pi*y)
    ]

    mt = MersenneTwister(4321) # use a different seed than the check!
    points = vcat([0,0,0,1,1,0,1,1], rand(mt,30*30*2))

    for (j, func) in enumerate(funcs)
        # scipy can get away with atol=0.22 here ...
        check_accuracy(func; x=points, intrp_tol=1e-9, atol=5e-3, rtol=1e-2)
        check_accuracy(func; x=points, intrp_tol=1e-9, atol=5e-3, rtol=1e-2, rescale=true)
    end

    ### scipy/interpolate/tests/test_interpnd: TestCloughTocher2DInterpolator::test_boundary_tri_symmetry
    # Interpolation at neighbourless triangles should retain
    # symmetry with mirroring the triangle.

    # Equilateral triangle
    points = [0,0,1,0,0.5,sqrt(3)/2]
    values = [1,0,0]

    ip = CloughTocher2DInterpolator(points, values)
    # Set gradient to zero at vertices
    fill!(ip.grad, 0.0)

    # Interpolation should be symmetric vs. bisector
    alpha = 0.3
    p1 = [0.5*cos(alpha), 0.5*sin(alpha)]
    p2 = [0.5*cos(pi/3-alpha), 0.5*sin(pi/3-alpha)]

    v1 = ip(p1)
    v2 = ip(p2)
    @assert v1 ≈ v2

    # ... and affine invariant
    mt = MersenneTwister(1)
    A = rand(mt,(2,2))
    b = rand(mt,2)

    points = reduce(vcat, [ A * pt .+ b for pt in eachcol(reshape(points,2,3)) ])
    p1 = A * p1 .+ b
    p2 = A * p2 .+ b

    ip = CloughTocher2DInterpolator(points, values)
    fill!(ip.grad, 0.0)

    w1 = ip(p1)
    w2 = ip(p2)

    @assert w1 ≈ v1
    @assert w2 ≈ v2

end
