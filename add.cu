#include <iostream>
using namespace std;

__global__
void add(
        float *x,
        float *y,
        float *res,
        int n
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
        res[i] = x[i] + y[i];
}

int main()
{
    int N = 1 << 26; // 1M elements
    float *x, *y, *res;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&res, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add << < numBlocks, blockSize >> > (x, y, res, N);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(res[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

// Free memory
//    delete[] x;
//    delete[] y;
//    delete[] res;
//
    cudaFree(x);
    cudaFree(y);
    cudaFree(res);
    return 0;
}