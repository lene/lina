//
// Created by lene on 04.05.15.
//

#include "RandomFiller.h"

template<> bool RandomFiller<double>::_inited = false;
template<> bool RandomFiller<float>::_inited = false;

template<>
double RandomFiller<double>::random() {
    init();
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

template<>
float RandomFiller<float>::random() {
    return static_cast<float>(RandomFiller<double>::random());
}

