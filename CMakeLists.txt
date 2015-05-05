cmake_minimum_required(VERSION 3.2)
project(lina)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
MESSAGE(WARNING "Include:  " ${GTEST_INCLUDE_DIRS})
MESSAGE(WARNING "Libs: " ${GTEST_LIBRARIES})

# Locate OpenCl
find_package(OpenCL REQUIRED)
MESSAGE(WARNING "Found: " ${OPENCL_FOUND})
if (OPENCL_FOUND)
    include_directories(${OPENCL_INCLUDE_DIRS})
    set(LIBS "${LIBS} ${OPENCL_LIBRARIES}")
endif (OPENCL_FOUND)
MESSAGE(WARNING "Include:  " ${OPENCL_INCLUDE_DIRS})
MESSAGE(WARNING "Libs: " ${OPENCL_LIBRARIES})
# fuck this. none of my attempts to find the OpenCl libs work.
set(LIBS "${LIBS} -L/opt/AMDAPPSDK-2.9-1/lib/x86_64 -lOpenCL")

# Link lina_test with what we want to test and the GTest and pthread library
add_executable(lina_test test/tests.cpp)
target_link_libraries(lina_test ${GTEST_LIBRARIES} pthread)

set(SOURCE_FILES src/main.cpp src/Timer.cpp src/Timer.h src/RandomFiller.cpp src/RandomFiller.h src/BenchmarkVienna.impl.h src/BenchmarkVienna.h src/CostFunction.impl.h src/CostFunction.h src/FileReader.impl.h src/FileReader.h src/MatrixPrinter.h  src/VectorPrinter.h src/FeatureNormalize.cpp src/FeatureNormalize.h)
add_executable(lina ${SOURCE_FILES})
# also, fuck the attempt above. in fact, fuck CMake.
target_link_libraries (lina -L/opt/AMDAPPSDK-2.9-1/lib/x86_64 -lOpenCL)

add_custom_target(
    run_tests
    lina_test
)
add_dependencies(lina run_tests)
