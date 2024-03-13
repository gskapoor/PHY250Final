// Matrix.h

#define CL_TARGET_OPENCL_VERSION 120

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class Matrix {
public:
    // Constructors
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, const std::vector<std::vector<int>>& data);

    // Destructor
    ~Matrix();

    // Copy constructor
    Matrix(const Matrix& other);

    // Assignment operator
    Matrix& operator=(const Matrix& other);

    // Accessor methods
    int getRows() const;
    int getCols() const;
    int getElement(int row, int col) const;

    // Mutator methods
    void setElement(int row, int col, int value);

    // Matrix operations
    Matrix transpose() const;
    Matrix multiplyCPU(Matrix& other) ;
    Matrix multiplyOpenCL(Matrix& other);

    // Print matrix
    void print() const;

private:
    int rows;
    int cols;
    std::vector<std::vector<int>> data;

    // OpenCL variables
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel;

    // OpenCL buffers
    cl_mem bufferA;
    cl_mem bufferB;
    cl_mem bufferResult;

    cl_command_queue commandQueue;

    // OpenCL initialization
    void initializeOpenCL();
};

#endif // MATRIX_H

