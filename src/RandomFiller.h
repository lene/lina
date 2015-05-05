//
// Created by lene on 04.05.15.
//

#ifndef TAVSIYE_RANDOMFILLER_H
#define TAVSIYE_RANDOMFILLER_H

#include <boost/numeric/ublas/matrix.hpp>


#include <cstddef>
#include <cstdlib>
#include <sys/time.h>

template <typename NUM>
class RandomFiller {

public:
    static void init() {
        if (_inited) return;
        srand( (unsigned int)time(NULL) );
    }

    static NUM random();

    template <typename orientation>
    static void setup_matrix(boost::numeric::ublas::matrix<NUM, orientation> &M) {
        for (unsigned int i = 0; i < M.size1(); ++i)
            for (unsigned int j = 0; j < M.size2(); ++j)
                M(i,j) = random();
    }

private:
    static bool _inited;

};


#endif //TAVSIYE_RANDOMFILLER_H
