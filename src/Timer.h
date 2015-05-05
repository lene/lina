//
// Created by lene on 04.05.15.
//

#ifndef TAVSIYE_TIMER_H
#define TAVSIYE_TIMER_H

class Timer {

public:

    Timer() : ts(0) {}

    void start();
    double get() const;

private:
    double ts;
};



#endif //TAVSIYE_TIMER_H
