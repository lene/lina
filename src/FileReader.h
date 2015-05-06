//
// Created by lene on 04.05.15.
//

#ifndef TAVSIYE_FILEREADER_H
#define TAVSIYE_FILEREADER_H


#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

namespace FileReader {

    template<typename Scalar> viennacl::matrix<Scalar> read_matrix(const std::string &filename);
    template<typename Scalar> viennacl::matrix<Scalar> read_matrix(std::istream &in);
    template<typename Scalar> viennacl::vector<Scalar> read_vector(const std::string &filename);
    template<typename Scalar> viennacl::vector<Scalar> read_vector(std::istream &in);

    const std::string testmatrix = "3 47 \
2104 1600 2400 1416 3000 1985 1534 1427 1380 1494 1940 2000 1890 4478 1268 2300 1320 1236 2609 3031 1767 1888 1604 1962 3890 1100 1458 2526 2200 2637 1839 1000 2040 3137 1811 1437 1239 2132 4215 2162 1664 2238 2567 1200 852 1852 1203 \
3 3 3 2 4 4 3 3 3 3 4 3 3 5 3 4 2 3 4 4 3 2 3 4 3 3 3 3 3 3 2 1 4 3 4 3 3 4 4 4 2 3 4 3 2 4 3";
    const std::string testvector ="47 \
399900 329900 369000 232000 539900 299900 314900 198999 212000 242500 239999 347000 329999 699900 259900 449900 299900 199900 499998 599000 252900 255000 242900 259900 573900 249900 464500 469000 475000 299900 349900 169900 314900 579900 285900 249900 229900 345000 549000 287000 368500 329900 314000 299000 179900 299900 239500";

};

#include "FileReader.impl.h"

#endif //TAVSIYE_FILEREADER_H
