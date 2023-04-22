# CloughTocher2DInterpolation

__Shamless and direct clone of scipy's CloughTocher2DInterpolator for Julia__

In particular, this package implements 
(`scipy.interpolate.CloughTocher2DInterpolator`)[https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html].

## Usage

```julia
using CloughTocher2DInterpolation

points         = [0,0, 0,1, 1,0, 1,1, 2,0, 2,1]
ipoints        = [0.5,0.5, 1.5,0.5]
real_values    = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
complex_values = [1.0+2.0im, 2.0+3.0im, 3.0-1.0im, 4.0-0.5im, 5.0-3.0im, 6.0+5.0im]

ip = CloughTocher2DInterpolator(points, real_values)
ip(ipoints)
ip = CloughTocher2DInterpolator(points, complex_values)
ip(ipoints)
```

## Acknowledgement

To make this work we ported/cloned/copied parts of the following files from the
(`scipy`)[https://github.com/scipy/scipy] repo
((commit https://github.com/scipy/scipy/tree/445777598fc0e2cb96d9a506acb32b1d2655f80c)[https://github.com/scipy/scipy/tree/445777598fc0e2cb96d9a506acb32b1d2655f80c]):
- `scipy/spatial/_qhull.pyx`
- `scipy/interpolate/interpnd.pyx`
- `scipy/spatial/tests/test__qhull.pyx`
- `scipy/interpolate/tests/test_interpnd.pyx`
