//
// Created by lene on 04.05.15.
//

#include "BenchmarkVienna.h"

#include "Timer.h"
#include "RandomFiller.h"

#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric;

// g++ cost_function.cpp -o cost_function -lOpenCL
#define VIENNACL_WITH_OPENCL 1

// Must be set if you want to use ViennaCL algorithms on ublas objects
// g++ cost_function.cpp -o cost_function
#define VIENNACL_WITH_UBLAS 1

#ifndef NDEBUG
#define NDEBUG
#endif

// ViennaCL headers
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"


template <typename ScalarType>
int BenchmarkVienna<ScalarType>::run(unsigned matrix_size) {
    Timer timer;
    double exec_time;
    ublas::matrix<ScalarType> ublas_A(matrix_size, matrix_size);
    ublas::matrix<ScalarType, ublas::column_major> ublas_B(matrix_size, matrix_size);
    ublas::matrix<ScalarType> ublas_C(matrix_size, matrix_size);
    ublas::matrix<ScalarType> ublas_C1(matrix_size, matrix_size);
    RandomFiller<ScalarType>::setup_matrix(ublas_A);
    RandomFiller<ScalarType>::setup_matrix(ublas_B);

    //viennacl::ocl::set_context_device_type(0, viennacl::ocl::gpu_tag());  //uncomment this is you wish to use GPUs only
    viennacl::matrix<ScalarType> vcl_A(matrix_size, matrix_size);
    viennacl::matrix<ScalarType, viennacl::column_major> vcl_B(matrix_size, matrix_size);
    viennacl::matrix<ScalarType> vcl_C(matrix_size, matrix_size);

    std::cout << "--- Computing matrix-matrix product using ublas ---" << std::endl;
    timer.start();
    ublas_C = ublas::prod(ublas_A, ublas_B);
    exec_time = timer.get();
    std::cout << " - Execution time: " << exec_time << std::endl;

    std::cout << std::endl << "--- Computing matrix-matrix product on each available compute device using ViennaCL ---" << std::endl;

    std::vector<viennacl::ocl::device> devices = viennacl::ocl::current_context().devices();
    for (std::size_t device_id=0; device_id<devices.size(); ++device_id)
    {
        viennacl::ocl::current_context().switch_device(devices[device_id]);
        std::cout << " - Device Name: " << viennacl::ocl::current_device().name() << std::endl;

        viennacl::copy(ublas_A, vcl_A);
        viennacl::copy(ublas_B, vcl_B);

        // ensure relevant OpenCL kernel is loaded before benchmarking
        vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
        viennacl::backend::finish();

        timer.start();
        vcl_C = viennacl::linalg::prod(vcl_A, vcl_B);
        viennacl::backend::finish();
        exec_time = timer.get();
        std::cout << " - Execution time on device (no setup time included): " << exec_time << std::endl;
        viennacl::copy(vcl_C, ublas_C1);
        std::cout << " - Checking result... ";
        bool check_ok = is_equal(ublas_C1, ublas_C);
        if (check_ok)
            std::cout << "[OK]" << std::endl << std::endl;
        else
            std::cout << "[FAILED]" << std::endl << std::endl;
    }

    return EXIT_SUCCESS;
}

template <typename ScalarType>
template <typename orientation>
bool BenchmarkVienna<ScalarType>::is_equal(const ublas::matrix<ScalarType, orientation> &A,
              const ublas::matrix<ScalarType, orientation> &B) {
    for (std::size_t i = 0; i < A.size1(); ++i) {
        for (std::size_t j = 0; j < A.size2(); ++j) {
            if ( std::fabs(A(i,j) - B(i,j)) / B(i,j) > 1e-4 ) return false;
        }
    }
    return true;
}
