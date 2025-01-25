# ds-cpp
C++ library for mathematical algorithms.  

# Current feature support
- Numerical Methods (extension `numerics`)
  - Solver for initial value problems of (stiff) ordinary differential equations (ode.h)
    - Unified class for solvers (ODESolver)
    - Explicit Euler Method (ExplicitEuler.h)
    - Explicit 5 step Runge-Kutta-Method (ode45.h)
    - Trapezoid rule for odes (odeTrapez.h)
    - Backward differential formula (odeBDF2.h)
  - Solver for systems of linear equations (gaussSeidel.h)
  - Gauss-Jordan method to calculate inverse matrices (gaussJordan.h)
  - QR-Decomposition of matrices (qr.h)
  - Singular Value Decomposition (SVD) (svd.h)
  - Fractals using numerical approximations (Fractals.h)
    - NewtonFractal
    - Mandelbrot
  - Newton method to approximate the zero-value for a given function based on an initial value newton.h
  - Function Interpolation/Approximation
    - 1D Interpolation
      - Polynomial Interpolation (PolynomialBase)
        - MonomBase
        - LagrangeBase
        - NewtonBase
      - Spline Interpolation
        - Spline: implements Natural cubic spline, as well as a B-Spline capable of interpolating 3D values 
    - 2D/3D Interpolation
      - see Spline
  - Differential calculus (Differentiation.h)
  - Numerical Integration (Integration.h)
- (classic) Statistics:
  - Probability.h
  - Insurance.h
- Plot support (uses/requires gnuplot see include/math/visualization/README.md or Plot.h)
- Data Science:
  - Classification:
    - NCC: Nearest Centroid Classifier (linear classifier) (NCC.h)
    - KNN: K Nearest Neighbor Classifier (non-linear classifier) (KNN.h)
  - Neural networks
    - Feed Forward NNs
      - Perceptron classifier (`Perceptron`)
      - Adaline Neuron Classifier
        - using gradient decent method (`AdalineGD`)
        - using statistics gradient decent method (`AdalineSGD`)

#### Build requirements
`gcc, cmake, clang`

# Installation

1. Download repository
`git clone git@github.com:/philsupertramp/ds.cpp`

2. Integration into your project

**Only CMake fully tested**
### cmake:
in your `CMakeLists.txt` add
```cmake
link_libraries(math-lib)
include_directories(math/include)
```

# Usage

After linking the library correctly one is able to include
the main Matrix class
```c++
#include <math/Matrix.h>
```
Specific implementations of algorithms are nested in subdirectories and can be imported on demand
```c++
#include <math/ds/KNN.h>
```

# Extensions
The library is split into multiple content seperated module.
Each module is nested in the root directory [`math/`](/include/math).  
Each sub-directory contains a README file with instructions how to use the module
and if it has dependencies.

### Adding extensions to library
To activate modules use the `MATH_EXTENSIONS` compile argument.

Example build command adding `numerics` module and building test suite:
```
mkdir -p build
cmake -DMATH_EXTENSIONS=numerics --build math/build -- -j 3 .
```

# Development
Feel free to contribute to the project!

### How to contribute
- first time contributors
    - fork the project
    - create a PR based on your repository
- known contributors
    - create PR based on `master` branch

**NOTE: please run `make lint` prior to submitting your code!**  
This requires `clang-format>=11.0.0`

# License
The project is under `MIT`-License see `LICENSE` for more

## Acknowledgements

