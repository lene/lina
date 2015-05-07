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
            out << std::setw(8) << std::setprecision(6) << vector_(i) << " ";
        }
        out << "|" << std::endl;
    }

private:
    const Vector &vector_;

};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const viennacl::vector<Scalar> &M) {
    VectorPrinter<viennacl::vector<Scalar>> p(M);
    p.print("", os);
    return os;
}


#endif //TAVSIYE_VECTORPRINTER_H
