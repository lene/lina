//
// Created by lene on 04.05.15.
//

#ifndef LINA_MATRIXPRINTER_H
#define LINA_MATRIXPRINTER_H

#include <iostream>
#include <iomanip>

#include <viennacl/matrix.hpp>

template <typename Matrix>
class MatrixPrinter {
public:
    MatrixPrinter(const Matrix &m): matrix_(m) {}

    void print(const std::string &msg = "", std::ostream &out = std::cout) const {
        for (unsigned i = 0; i < matrix_.size1(); ++i) {
            out << msg << "|";
            for (int j = 0; j < matrix_.size2(); ++j) {
                out << std::setw(8) << std::setprecision(4) << matrix_(i, j) << " ";
            }
            out << "|" << std::endl;
        }
    }

private:
    const Matrix &matrix_;
};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const viennacl::matrix<Scalar> &M) {
    MatrixPrinter<viennacl::matrix<Scalar>> p(M);
    p.print("", os);
    return os;
}


#endif //LINA_MATRIXPRINTER_H
