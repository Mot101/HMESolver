#pragma once
#include <string>
#include <vector>
#include <complex>

bool trySolvePolynomial(
    const std::string& expr,
    std::vector<std::complex<double>>& roots,
    std::string& normalized,
    std::string& err
);