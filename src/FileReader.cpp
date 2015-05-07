//
// Created by lene on 07.05.15.
//

#include "FileReader.impl.h"

template viennacl::matrix<float> FileReader::read_matrix(const std::string &filename);
template viennacl::matrix<double> FileReader::read_matrix(const std::string &filename);

template viennacl::matrix<float> FileReader::read_matrix(std::istream &in);
template viennacl::matrix<double> FileReader::read_matrix(std::istream &in);

template viennacl::vector<float> FileReader::read_vector(const std::string &filename);
template viennacl::vector<double> FileReader::read_vector(const std::string &filename);

template viennacl::vector<float> FileReader::read_vector(std::istream &in);
template viennacl::vector<double> FileReader::read_vector(std::istream &in);

template viennacl::matrix<float> FileReader::add_bias_column(const viennacl::matrix<float> &M);
template viennacl::matrix<double> FileReader::add_bias_column(const viennacl::matrix<double> &M);

