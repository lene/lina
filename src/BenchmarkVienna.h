//
// Created by lene on 04.05.15.
//

#ifndef LINA_BENCHMARKVIENNA_H
#define LINA_BENCHMARKVIENNA_H


#include "Timer.h"
#include "RandomFiller.h"

#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric;

template <typename ScalarType>
struct BenchmarkVienna {

    int run(unsigned matrix_size);

private:
    template <typename orientation>
    bool is_equal(const ublas::matrix<ScalarType, orientation> &A,
                  const ublas::matrix<ScalarType, orientation> &B);

};

#include "BenchmarkVienna.impl.h"

#endif //LINA_BENCHMARKVIENNA_H
