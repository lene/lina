// g++ cost_function.cpp -o cost_function -lOpenCL
#define VIENNACL_WITH_OPENCL 1

// Must be set if you want to use ViennaCL algorithms on ublas objects
// g++ cost_function.cpp -o cost_function 
#define VIENNACL_WITH_UBLAS 1

#ifndef NDEBUG
 #define NDEBUG
#endif

#include <iostream>

// ublas headers
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>




// ViennaCL headers
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

//#include "Random.hpp"

#define BLAS3_MATRIX_SIZE   1000

using namespace boost::numeric;


#ifndef VIENNACL_WITH_OPENCL
  struct dummy
  {
    std::size_t size() const { return 1; }
  };
#endif

#include <sys/time.h>

struct Timer {
  Timer() : ts(0) {}
   
  void start() {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    ts = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);
  }
    
  double get() const {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    double end_time = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);
    
    return static_cast<double>(end_time-ts) / 1000000.0;
  }
    
private:
  double ts;   
};

inline void init()
{
  static bool init = false;
  if (!init)
  {
    srand( (unsigned int)time(NULL) );
    init = true;
  }
}
 
template<class TYPE>
TYPE random();
 
template<>
double random<double>()
{
  init();
  return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}
 
template<>
float random<float>()
{
  init();
  return static_cast<float>(random<double>());
}

template <typename ScalarType, typename orientation>
void setup_matrix(ublas::matrix<ScalarType, orientation> &M) {
  for (unsigned int i = 0; i < M.size1(); ++i)
    for (unsigned int j = 0; j < M.size2(); ++j)
      M(i,j) = random<ScalarType>();
}

template <typename ScalarType, typename orientation>
bool is_equal(const ublas::matrix<ScalarType, orientation> &A, 
			  const ublas::matrix<ScalarType, orientation> &B) {
  for (std::size_t i = 0; i < A.size1(); ++i) {
    for (std::size_t j = 0; j < A.size2(); ++j) {
      if ( std::fabs(A(i,j) - B(i,j)) / B(i,j) > 1e-4 ) return false;
    }
  }
  return true;
}

int main()
{
  typedef double ScalarType;

  Timer timer;
  double exec_time;
  ublas::matrix<ScalarType> ublas_A(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  ublas::matrix<ScalarType, ublas::column_major> ublas_B(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  ublas::matrix<ScalarType> ublas_C(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  ublas::matrix<ScalarType> ublas_C1(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  setup_matrix(ublas_A);
  setup_matrix(ublas_B);

  //viennacl::ocl::set_context_device_type(0, viennacl::ocl::gpu_tag());  //uncomment this is you wish to use GPUs only
  viennacl::matrix<ScalarType> vcl_A(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  viennacl::matrix<ScalarType, viennacl::column_major> vcl_B(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);
  viennacl::matrix<ScalarType> vcl_C(BLAS3_MATRIX_SIZE, BLAS3_MATRIX_SIZE);

  std::cout << "--- Computing matrix-matrix product using ublas ---" << std::endl;
  timer.start();
  ublas_C = ublas::prod(ublas_A, ublas_B);
  exec_time = timer.get();
  std::cout << " - Execution time: " << exec_time << std::endl;

  std::cout << std::endl << "--- Computing matrix-matrix product on each available compute device using ViennaCL ---" << std::endl;

#ifdef VIENNACL_WITH_OPENCL
  std::vector<viennacl::ocl::device> devices = viennacl::ocl::current_context().devices();
#else
  dummy devices;
#endif
  for (std::size_t device_id=0; device_id<devices.size(); ++device_id)
  {
#ifdef VIENNACL_WITH_OPENCL
    viennacl::ocl::current_context().switch_device(devices[device_id]);
    std::cout << " - Device Name: " << viennacl::ocl::current_device().name() << std::endl;
#endif
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
