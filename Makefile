lint:
	find . -regex '.*\.\(cpp\|hpp\|cc\|cxx\|h\)' -not -path '*/pyDS/*' -exec clang-format -i {} \;

test:
	./build.sh cmake-debug-coverage --test -o="MATH_EXTENSIONS=ds,numerics,symb,stat";

test-release:
	./build.sh cmake-release-coverage --test -t=Release -o="MATH_EXTENSIONS=ds,numerics,symb,stat";

build-coverage:
	./build.sh cmake-debug-coverage;

coverage:
	./build.sh cmake-debug-coverage -c -o="MATH_EXTENSIONS=numerics,ds,stat,symb -DMATH_IMAGE_PROCESSING=1";
	lcov -c -d cmake-debug-coverage --rc lcov_branch_coverage=1 --rc lcov_function_coverage=1 -o test.info -f --include \*/include/math/\* --include \*/include/math/numerics\* --include \*/math/numerics/lin_alg\* --include \*/math/numerics/ode\* --include \*/math/statistics/\* --include \*/math/ds\*;
	lcov -a test.info -a base.info -o cov.info;
	genhtml --demangle-cpp --function-coverage --branch-coverage cov.info -o coverage-report;

test-coverage:
	./build.sh cmake-debug-coverage -c -o="MATH_EXTENSIONS=numerics,ds,vis,symb" --target=symb_TestSymbolic;
	lcov -c -d cmake-debug-coverage --rc lcov_branch_coverage=1 -o test.info -f --include \*/include/math/\* --include \*/include/math/numerics\* --include \*/math/numerics/lin_alg\* --include \*/math/numerics/ode\*;
	lcov -a test.info -a base.info -o cov.info;
	genhtml --branch-coverage cov.info -o coverage-report;

minimal:
	./build.sh cmake-release-minimal -t=Release;

image-processing:
	./build.sh cmake-debug-coverage -c -o="MATH_EXTENSIONS=ds,numerics" -o="MATH_IMAGE_PROCESSING=1";

benchmark:
	./build.sh cmake-release -b -o="MATH_EXTENSIONS=numerics" --target=benchmarks -t=Release;
