cmake_minimum_required(VERSION 3.16)
project(superfaiss CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(superfaiss STATIC
	src/kernels.cpp
	src/kernels_avx2.cpp
	src/alloc.cpp
	src/validate.cpp
	src/bake.cpp
	src/query.cpp
	src/compose.cpp
	src/analytics.cpp
	src/pca.cpp
	src/scratch.cpp
	src/graph.cpp
	src/novelty.cpp
	src/matching.cpp
	src/diversity.cpp
)
target_include_directories(superfaiss PUBLIC include)

# Three compiler families, told apart by the compiler ID and its command-line frontend.
# CMake sets MSVC for clang-cl too (Clang with the MSVC-compatible frontend), so a bare
# if(MSVC) sends clang-cl down the cl.exe path and it never gets the flags clang needs.
#   - cl.exe:   MSVC switches only.
#   - clang-cl: MSVC switches, plus the clang flags it cannot infer from them (below).
#   - GCC/Clang with the GNU frontend: GNU switches.
set(SUPERFAISS_CLANG_CL OFF)
if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
		AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
	set(SUPERFAISS_CLANG_CL ON)
endif()

# Determinism: forbid FP contraction (implicit fused multiply-add would change rounding
# and break scalar/SIMD bit-equality; the AVX2 path uses FMA *explicitly*, mirrored by
# std::fma). cl.exe's /fp:precise does not contract; clang-cl maps /fp:precise to
# clang's precise model, which does (-ffp-contract=on), so clang-cl also gets
# -ffp-contract=off. /clang: passes an option to the clang driver verbatim.
set(SUPERFAISS_CLANG_CL_OPTIONS)
if(SUPERFAISS_CLANG_CL)
	set(SUPERFAISS_CLANG_CL_OPTIONS /clang:-ffp-contract=off)
endif()

if(MSVC)
	target_compile_options(superfaiss PRIVATE /W4 /fp:precise ${SUPERFAISS_CLANG_CL_OPTIONS})
	# VEX-encode the whole AVX2 TU so no SSE/AVX transition penalties occur. The
	# functions are runtime-dispatched; the TU is never executed on non-AVX2 CPUs.
	set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
	# cl.exe compiles SSE4.1 intrinsics on the x64 baseline without a flag; clang-cl, like
	# clang, needs the target feature enabled for the TU, or it rejects them. Keyed on the
	# target architecture the compiler reports (x64 or X86), not the host processor.
	if(SUPERFAISS_CLANG_CL AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^(x64|X86)$")
		set_source_files_properties(src/kernels.cpp PROPERTIES COMPILE_OPTIONS "/clang:-msse4.2")
	endif()
else()
	target_compile_options(superfaiss PRIVATE -Wall -Wextra -ffp-contract=off)
	# GCC/Clang default to the SSE2 baseline on x86; the SSE kernels are SSE4.1.
	# (The AVX2 TU needs no flags — its functions carry target("avx2,fma") attributes.)
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
		set_source_files_properties(src/kernels.cpp PROPERTIES COMPILE_OPTIONS "-msse4.2")
	endif()
endif()

add_executable(superfaiss_tests tests/test_main.cpp)
find_package(Threads REQUIRED)
target_link_libraries(superfaiss_tests PRIVATE superfaiss Threads::Threads)
if(MSVC)
	target_compile_options(superfaiss_tests PRIVATE /W4 /fp:precise ${SUPERFAISS_CLANG_CL_OPTIONS})
else()
	target_compile_options(superfaiss_tests PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

add_executable(superfaiss_bench tests/bench_main.cpp)
target_link_libraries(superfaiss_bench PRIVATE superfaiss Threads::Threads)
if(MSVC)
	target_compile_options(superfaiss_bench PRIVATE /W4 /fp:precise ${SUPERFAISS_CLANG_CL_OPTIONS})
else()
	target_compile_options(superfaiss_bench PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

enable_testing()
add_test(NAME superfaiss_tests COMMAND superfaiss_tests)
