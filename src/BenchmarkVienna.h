//
// Created by lene on 04.05.15.
//

#ifndef LINA_BENCHMARKVIENNA_H
#define LINA_BENCHMARKVIENNA_H


#include "Timer.h"
#include "RandomFiller.h"

#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric;

template <typename Scalar>
struct BenchmarkVienna {

    static_assert(std::is_floating_point<Scalar>::value, "Scalar is not a floating point type");

    int run(unsigned matrix_size);

private:
    template <typename orientation>
    bool is_equal(const ublas::matrix<Scalar, orientation> &A,
                  const ublas::matrix<Scalar, orientation> &B);

};

#endif //LINA_BENCHMARKVIENNA_H
