// matrix.cpp
// Gobind Kapoor, Jovin Louie
// 2023-03-13
//

#include "matrix.hpp"

const char* kernelSource = R"(
    __kernel void matrixMul(__global const int* A,
                             __global const int* B,
                             __global int* C,
                             const int rowsA,
                             const int colsA,
                             const int colsB) {
        int globalRow = get_global_id(0);
        int globalCol = get_global_id(1);
        int sum = 0;

        for (int k = 0; k < colsA; ++k) {
            sum += A[globalRow * colsA + k] * B[k * colsB + globalCol];
        }

        C[globalRow * colsB + globalCol] = sum;
    }
)";

Matrix::Matrix() : rows(0), cols(0), platform(nullptr), device(nullptr), context(nullptr), program(nullptr), kernel(nullptr),
                   bufferA(nullptr), bufferB(nullptr), bufferResult(nullptr) {
    initializeOpenCL();
}

Matrix::Matrix(int rows, int cols) : Matrix() {
    this->rows = rows;
    this->cols = cols;
    data = std::vector<std::vector<int>>(rows, std::vector<int>(cols, 0));
}

Matrix::Matrix(int rows, int cols, const std::vector<std::vector<int>>& data) : Matrix(rows, cols) {
    this->data = data;
}

Matrix::~Matrix() {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
}

Matrix::Matrix(const Matrix& other) : Matrix() {
    rows = other.rows;
    cols = other.cols;
    data = other.data;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

int Matrix::getRows() const {
    return rows;
}

int Matrix::getCols() const {
    return cols;
}

int Matrix::getElement(int row, int col) const {
    return data[row][col];
}

void Matrix::setElement(int row, int col, int value) {
    data[row][col] = value;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[j][i] = data[i][j];
        }
    }

    return result;
}

Matrix Matrix::multiplyCPU(Matrix& other){
    if (cols != other.rows) {
        return Matrix();
    }

    Matrix result(rows, other.cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            int sum = 0;
            for (int k = 0; k < cols; ++k) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }

    return result;
}

void Matrix::initializeOpenCL() {
    cl_uint numPlatforms;

    cl_int error;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    if (numPlatforms == 0) {
        std::cerr << "Error: No OpenCL platforms available." << std::endl;
        return;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    error = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "ERROR GETTING PLATFORM ID" << std::endl;
    }

    platform = platforms[0];

    cl_uint numDevices;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
    if (error != CL_SUCCESS) {
      std::cerr << "ERROR GETTING Device IDS" << std::endl;
    }

    if (numDevices == 0) {
        std::cerr << "Error: No GPU devices available." << std::endl;
        return;
    }

    std::vector<cl_device_id> devices(numDevices);
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "ERROR GETTING Device IDS" << std::endl;
    }


    device = devices[0];

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

    commandQueue = clCreateCommandQueue(context, device, 0, NULL);


    program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      std::cerr << "ERROR Building program" << std::endl;
    }

    kernel = clCreateKernel(program, "matrixMul", nullptr);


}

Matrix Matrix::multiplyOpenCL(Matrix& other){
    if (context == nullptr || device == nullptr || program == nullptr || kernel == nullptr) {
        return Matrix();
    }

    if (cols != other.rows) {
        return Matrix();
    }

    std::vector<int> tempBufferA;
    for (const auto& row : data) {
        tempBufferA.insert(tempBufferA.end(), row.begin(), row.end());
    }

    std::vector<int> tempBufferB;
    for (const auto& row : other.data) {
        tempBufferB.insert(tempBufferB.end(), row.begin(), row.end());
    }

    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(int) * rows * cols, tempBufferA.data(), nullptr);

    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(int) * other.rows * other.cols, tempBufferB.data(), nullptr);

    bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  sizeof(int) * rows * other.cols, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 3, sizeof(int), &rows);
    clSetKernelArg(kernel, 4, sizeof(int), &cols);
    clSetKernelArg(kernel, 5, sizeof(int), &other.cols);

    size_t globalWorkSize[2] = {static_cast<size_t>(rows), static_cast<size_t>(other.cols)};
    size_t localWorkSize[2] = {1, 1}; // Adjust as needed

    clEnqueueNDRangeKernel(commandQueue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    clFinish(commandQueue);

    std::vector<int> resultData(rows * other.cols);
    clEnqueueReadBuffer(commandQueue, bufferResult, CL_TRUE, 0, sizeof(int) * rows * other.cols, resultData.data(), 0, nullptr, nullptr);

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);

    std::vector<std::vector<int>> resultMatrix(rows, std::vector<int>(other.cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            resultMatrix[i][j] = resultData[i * other.cols + j];
        }
    }

    return Matrix(rows, other.cols, resultMatrix);
}

void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << "\n";
    }
}

