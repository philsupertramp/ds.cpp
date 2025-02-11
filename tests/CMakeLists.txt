cmake_minimum_required(VERSION 3.9)
project(math-tests)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#
#OPTION(ENABLE_GCOV "Enable gcov (debug, Linux builds only)" OFF)

find_program(CTEST_COVERAGE_COMMAND NAMES gcov)
include(CTest)
enable_testing()

include_directories(../include)

IF (MATH_COVERAGE)
    string(APPEND CMAKE_C_FLAGS " -fprofile-arcs -ftest-coverage -DHAVE_GCOV")
    string(APPEND CMAKE_CXX_FLAGS " -fprofile-arcs -ftest-coverage -DHAVE_GCOV")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -fprofile-arcs -ftest-coverage -lgcov")
ENDIF()
IF(DEBUG)
    string(APPEND CMAKE_C_FLAGS " -fsanitize=address")
    string(APPEND CMAKE_CXX_FLAGS " -fsanitize=address")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address")

ENDIF()

function(add_test_source SOURCE_NAME)
    string(REPLACE ".cpp" "" NAME ${SOURCE_NAME})
    string(REPLACE "/" "_" NAME ${NAME})
    message("building ${NAME} ${SOURCE_NAME}")
    add_executable(${NAME} ${SOURCE_NAME})
    target_link_libraries(${NAME} math-lib)
    add_test(${NAME} ${NAME})
endfunction()


add_test_source(TestMatrix.cpp)
add_test_source(dl/nn/TestEngine.cpp)
add_test_source(dl/nn/TestNN.cpp)
add_test_source(dl/nn/TestMatrix.cpp)

if (MATH_EXTENSIONS MATCHES "(numerics)")
    add_test_source(numerics/TestUtils.cpp)
    add_test_source(numerics/TestFractals.cpp)

    add_test_source(numerics/ode/TestODE45.cpp)
    add_test_source(numerics/ode/TestExplicitEuler.cpp)
    add_test_source(numerics/ode/TestODETrapez.cpp)
    add_test_source(numerics/ode/TestODEBDF2.cpp)

    add_test_source(numerics/lin_alg/TestBackwardSub.cpp)
    add_test_source(numerics/lin_alg/TestForwardSub.cpp)
    add_test_source(numerics/lin_alg/TestGaussSeidel.cpp)
    add_test_source(numerics/lin_alg/TestNewton.cpp)
    add_test_source(numerics/lin_alg/TestLU.cpp)
    add_test_source(numerics/lin_alg/TestQR.cpp)
    add_test_source(numerics/lin_alg/TestSVD.cpp)

    add_test_source(numerics/analysis/TestSupportValues.cpp)
    add_test_source(numerics/analysis/TestNaturalSpline.cpp)
    add_test_source(numerics/analysis/TestDifferentiation.cpp)
    add_test_source(numerics/analysis/TestIntegration.cpp)
endif()

if(MATH_EXTENSIONS MATCHES "(symb)")
    add_test_source(symb/TestSymbolic.cpp)
endif()

if (MATH_EXTENSIONS MATCHES "(vis)")
    add_compile_definitions(USE_VIS)
    add_test_source(plot/TestPlot.cpp)
endif()

if (MATH_EXTENSIONS MATCHES "(stat)")
    add_test_source(statistics/TestProbability.cpp)
    add_test_source(statistics/TestInsurance.cpp)
endif()

if (MATH_EXTENSIONS MATCHES "(ds)")
    add_test_source(ds/TestDSUtils.cpp)
    add_test_source(ds/TestPerceptron.cpp)
    add_test_source(ds/TestAdalineGD.cpp)
    add_test_source(ds/TestAdalineSGD.cpp)
    add_test_source(ds/TestSGD.cpp)
    add_test_source(ds/TestLogRegSGD.cpp)
    add_test_source(ds/TestNCC.cpp)
    add_test_source(ds/TestKNN.cpp)
    add_test_source(ds/TestDecisionTree.cpp)
    add_test_source(ds/preprocessing/TestScaler.cpp)
    add_test_source(ds/decomposition/TestPCA.cpp)
endif()

IF (MATH_COVERAGE)
    set(TARGET_NAME run_tests)
    add_custom_target(${TARGET_NAME}
            COMMAND "ctest -VV" )

ENDIF()
