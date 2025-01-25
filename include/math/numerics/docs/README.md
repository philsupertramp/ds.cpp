# numerics
split into submodules

## /lin_alg/  
Contains newton-algorithm and Gauss-Seidel-Algorithm to solve systems of linear equations.  
Also contains LU-substitution.

Newton-Fractals:  
![newton-fractal](https://user-images.githubusercontent.com/9550040/139021410-708a7676-c06f-418c-8edb-58a50f667607.png)

Mandelbrot set:
![mandelbrot](https://user-images.githubusercontent.com/9550040/120936014-eecb5900-c705-11eb-918a-607088ce6d87.png)

To learn more about these beautiful fractals check out 3b1b video  
https://www.youtube.com/watch?v=-RdOwhmqP5s

## /ode/  
ODESolver.h contains several solvers for ordinary differential equations.    
Hint! If you wanna approximate a pde (partial differential equation) just transform it into a system of ode's and use
BDF(2) ;)

## /analysis/

Contains `SupportValues.h` and `Spline.h`  
both methods to interpolate a function by given support values.  

Famous example by runge  

![spline-interpol-runge](https://user-images.githubusercontent.com/9550040/118409523-1ee58600-b68b-11eb-863e-5ed136a24bb4.png)
