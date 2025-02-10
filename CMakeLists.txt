cmake_minimum_required(VERSION 3.9)
project(math-lib CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-DMATH_SILENCE_WARNING)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_definitions(-DCOMPILER_CLANG)
    message("CLANG")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_definitions(-DCOMPILER_GCC)
#    set(CMAKE_CXX_FLAGS -fprofile-arcs -ftest-coverage)
    set(CMAKE_LFLAGS --coverage)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_definitions(-DCOMPILER_VC)
endif()

set(LIB_HEADERS
      include/math/Matrix.h
      include/math/format.h
      include/math/Random.h
      include/math/sorting.h
      include/math/matrix_utils.h
    include/math/dl/nn/engine.h
    include/math/dl/nn/nn.h
)

set(LIB_SOURCES

    src/Matrix.cpp
    src/format.cpp
)
if(MATH_EXTENSIONS MATCHES "(numerics)")
    set(LIB_HEADERS
            ${LIB_HEADERS}
            include/math/numerics/utils.h
            include/math/numerics/lin_alg/newton.h
            include/math/numerics/lin_alg/backwardSub.h
            include/math/numerics/lin_alg/forwardSub.h
            include/math/numerics/lin_alg/LU.h
            include/math/numerics/lin_alg/gaussSeidel.h
            include/math/numerics/ode.h
            include/math/numerics/ode/ODESolver.h
            include/math/numerics/ode/ExplicitEuler.h
            include/math/numerics/ode/ode.h
            include/math/numerics/ode/odeTrapez.h
            include/math/numerics/ode/ode45.h
            include/math/numerics/ode/odeBDF2.h

            include/math/numerics/analysis/SupportValues.h
            include/math/numerics/analysis/Spline.h
            include/math/numerics/analysis/Differentiation.h
            include/math/numerics/Fractals.h
            include/math/numerics/lin_alg/gaussJordan.h
            include/math/numerics/lin_alg/qr.h
            include/math/numerics/lin_alg/svd.h
    )
    set(LIB_SOURCES
            ${LIB_SOURCES}
            src/numerics/utils.cpp
            )
endif()
if(MATH_EXTENSIONS MATCHES "(symb)")
    set(LIB_HEADERS
        ${LIB_HEADERS}

        include/math/symb/Equation.h
            include/math/symb/MathNode.h
            include/math/symb/Operator.h
            include/math/symb/Operand.h
            include/math/symb/EquationParser.h
            )
    set(LIB_SOURCES
            ${LIB_SOURCES}
            src/symb/MathNode.cpp
            src/symb/Operand.cpp
            src/symb/Operator.cpp
            src/symb/EquationParser.cpp
            )
endif()
if (MATH_EXTENSIONS MATCHES "(ds)")
    set(LIB_HEADERS
        ${LIB_HEADERS}
            include/math/Matrix.h
            include/math/ds/DataSet.h
            include/math/ds/Classifier.h
            include/math/ds/Perceptron.h
            include/math/ds/AdalineGD.h
            include/math/ds/AdalineSGD.h
            include/math/ds/LogRegSGD.h
            include/math/ds/SGD.h
            include/math/ds/NCC.h
            include/math/ds/KNN.h
            include/math/ds/DecisionTree.h
            
            include/math/ds/preprocessing/Scaler.h
            include/math/ds/decomposition/PCA.h
            )
    if(MATH_IMAGE_PROCESSING)
        set(LIB_HEADERS ${LIB_HEADERS}
                include/math/ds/ImageDataSet.h
        )
        add_definitions( -DMAGICKCORE_QUANTUM_DEPTH=16 )
        add_definitions( -DMAGICKCORE_HDRI_ENABLE=0 )
        find_package(ImageMagick COMPONENTS Magick++)
        include_directories(${ImageMagick_INCLUDE_DIRS})
    endif()
endif ()

if (MATH_EXTENSIONS MATCHES "(vis)")
    set(LIB_HEADERS ${LIB_HEADERS}
            include/math/visualization/Plot.h
        )

    set(colorPallets gnpu magma inferno spectral)

    foreach(pallet IN LISTS colorPallets)
        configure_file(include/math/visualization/${pallet}.pal ${CMAKE_CURRENT_BINARY_DIR}/${pallet}.pal COPYONLY)
        configure_file(include/math/visualization/${pallet}.pal ${CMAKE_CURRENT_BINARY_DIR}/tests/${pallet}.pal COPYONLY)
    endforeach()

endif ()
if (MATH_EXTENSIONS MATCHES "(stat)")
    set(LIB_HEADERS ${LIB_HEADERS}
            include/math/statistics/Probability.h
            include/math/statistics/Insurance.h
            include/math/statistics/Finance.h
            )
    set(LIB_SOURCES
            ${LIB_SOURCES}
            src/statistics/Probability.cpp
            src/statistics/Insurance.cpp
            src/statistics/Finance.cpp
            )
endif ()

add_library(math-lib SHARED ${LIB_HEADERS} ${LIB_SOURCES})
set_target_properties(math-lib PROPERTIES LINKER_LANGUAGE CXX)


if (MATH_EXTENSIONS MATCHES "(ds)")
    target_link_libraries(math-lib ${ImageMagick_LIBRARIES})
endif()

# add_compile_options(-fsanitize=address -fsanitize=undefined)
# add_link_options(-fsanitize=address -fsanitize=undefined)

if(MATH_TESTS)
    include(CTest)
    set(MATH_COVERAGE 1)
    add_subdirectory(tests)
endif()
