using CloughTocher2DInterpolation

numpoints = 6
coordinates = [0,0,
               0,1,
               1,0,
               1,1,
               2,0,
               2,1]
values = Float64[1,2,3,4,5,6]
I = CloughTocher2DInterpolation.Interpolator(numpoints, coordinates, values)
icoords = [0.5, 0.5,
           1.5, 0.5]
I(icoords)
