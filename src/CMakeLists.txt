cmake_minimum_required(VERSION 3.2)
project(tavsiye)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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

set(SOURCE_FILES main.cpp)
add_executable(tavsiye ${SOURCE_FILES} Timer.cpp Timer.h RandomFiller.cpp RandomFiller.h BenchmarkVienna.impl.h BenchmarkVienna.h CostFunction.impl.h CostFunction.h FileReader.impl.h FileReader.h MatrixPrinter.h  VectorPrinter.h FeatureNormalize.cpp FeatureNormalize.h)
# also, fuck the attempt above. in fact, fuck CMake.
target_link_libraries (tavsiye -L/opt/AMDAPPSDK-2.9-1/lib/x86_64 -lOpenCL)