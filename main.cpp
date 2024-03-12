#include <vector>
#include <iostream>


#include "matrix.hpp"

int main() {
    // Example matrices
    std::vector<std::vector<int>> dataA = {{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<int>> dataB = {{7, 8}, {9, 10}, {11, 12}};

    Matrix matrixA(dataA.size(), dataA[0].size(), dataA);
    Matrix matrixB(dataB.size(), dataB[0].size(), dataB);

    // Print the original matrices
    std::cout << "Matrix A:" << std::endl;
    matrixA.print();
    std::cout << "Matrix B:" << std::endl;
    matrixB.print();

    // Matrix multiplication using CPU
    Matrix resultCPU = matrixA.multiplyCPU(matrixB);
    std::cout << "Result (CPU):" << std::endl;
    resultCPU.print();

    // Matrix multiplication using OpenCL
    Matrix resultOpenCL = matrixA.multiplyOpenCL(matrixB);
    std::cout << "Result (OpenCL):" << std::endl;
    resultOpenCL.print();

    return 0;
}

