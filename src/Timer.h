//
// Created by lene on 04.05.15.
//

#ifndef LINA_TIMER_H
#define LINA_TIMER_H

class Timer {

public:

    Timer() : ts(0) {}

    void start();
    double get() const;

private:
    double ts;
};



#endif //LINA_TIMER_H
