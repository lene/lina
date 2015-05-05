//
// Created by lene on 04.05.15.
//

#ifndef TAVSIYE_MATRIXPRINTER_H
#define TAVSIYE_MATRIXPRINTER_H

#include <iostream>
#include <iomanip>

template <typename Matrix>
class MatrixPrinter {
public:
    MatrixPrinter(const Matrix &m): matrix_(m) {}

    void print(std::ostream &out = std::cout) const {
        for (unsigned i = 0; i < matrix_.size1(); ++i) {
            out << "|";
            for (int j = 0; j < matrix_.size2(); ++j) {
                out << std::setw(8) << std::setprecision(4) << matrix_(i, j) << " ";
            }
            out << "|" << std::endl;
        }
    }

private:
    const Matrix &matrix_;
};


#endif //TAVSIYE_MATRIXPRINTER_H
