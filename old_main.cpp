#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
#include <vector>

// OpenCL version macro
#define OPENCL_VERSION_1_2

// Function to print the matrix
void printMatrix(const std::vector<int>& matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Matrix size (assuming square matrices)
    const int matrixSize = 3;

    // Input matrices
    std::vector<int> matrixA(matrixSize * matrixSize);
    std::vector<int> matrixB(matrixSize * matrixSize);

    // Fill matrices with random values
    for (int i = 0; i < matrixSize * matrixSize; ++i) {
        matrixA[i] = i;
        matrixB[i] = i;
    }

    // Result matrix
    std::vector<int> resultMatrix(matrixSize * matrixSize, 0);


    cl_int err;
    // OpenCL variables
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufferA, bufferB, bufferResult;

    // Initialize OpenCL
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS){
      std::cerr << "error getting platform id" << std::endl;
      return 1;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    if (err != CL_SUCCESS){
      std::cerr << "error getting device id" << std::endl;
      return 1;
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);


    // Create OpenCL buffers for matrices
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(int) * matrixSize * matrixSize, matrixA.data(), NULL);

    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(int) * matrixSize * matrixSize, matrixB.data(), NULL);

    bufferResult = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(int) * matrixSize * matrixSize, NULL, NULL);
    // Load OpenCL program source
    const char* kernelSource =
#ifdef OPENCL_VERSION_1_2
    R"(
        #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
        #pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
    )"
#endif
    R"(
        __kernel void matrixMul(__global const int* A,
                                 __global const int* B,
                                 __global int* C,
                                 const int size) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            int sum = 0;

            for (int i = 0; i < size; ++i) {
                sum += A[row * size + i] * B[i * size + col];
            }

            C[row * size + col] = sum;
        }
    )";

    // Create OpenCL program from source
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "matrixMul", NULL);

    // Set OpenCL kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 3, sizeof(int), &matrixSize);

    // Enqueue kernel for execution
    size_t globalWorkSize[2] = { matrixSize, matrixSize };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);

    // Read result from OpenCL buffer to host memory
    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0,
                        sizeof(int) * matrixSize * matrixSize, resultMatrix.data(), 0, NULL, NULL);

    // Print the matrices and the result
    std::cout << "Matrix A:" << std::endl;
    printMatrix(matrixA, matrixSize);

    std::cout << "Matrix B:" << std::endl;
    printMatrix(matrixB, matrixSize);

    std::cout << "Result Matrix:" << std::endl;
    printMatrix(resultMatrix, matrixSize);

    // Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

