//
// Created by lene on 04.05.15.
//

#include "Timer.h"

#include <cstddef>
#include <sys/time.h>

void Timer::start() {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    ts = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);
}

double Timer::get() const {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    double end_time = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);

    return static_cast<double>(end_time-ts) / 1000000.0;
}
