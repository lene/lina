//
// Created by lene on 04.05.15.
//

#ifndef TAVSIYE_VECTORPRINTER_H
#define TAVSIYE_VECTORPRINTER_H

#include <iostream>
#include <iomanip>

template <typename Vector>
class VectorPrinter {
public:
    VectorPrinter(const Vector &v): vector_(v) {}

    void print(const std::string &msg = "", std::ostream &out = std::cout) const {
        out << msg << " |";
        for (unsigned i = 0; i < vector_.size(); ++i) {
            out << std::setw(8) << std::setprecision(4) << vector_(i) << " ";
        }
        out << "|" << std::endl;
    }

private:
    const Vector &vector_;

};


#endif //TAVSIYE_VECTORPRINTER_H
