#include <iostream>

__global__
void add(
        const float *const x,
        const float *const y,
        float *const res,
        const int n
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        res[i] = x[i] + y[i];
}

void add_serial(
        const float *const x,
        const float *const y,
        float *const res,
        const int n
)
{
    for (int i = 0; i < n; ++i)
        res[i] = x[i] + y[i];
}

int main()
{
    int N = 1 << 28; // 1M elements
    float *x, *y, *res;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&res, N * sizeof(float));
//    x = new float[N];
//    y = new float[N];
//    res = new float[N];

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add <<< numBlocks, blockSize >>> (x, y, res, N);
    cudaDeviceSynchronize();
//    add_serial(x, y, res, N);

    std::cout << "Calc error...";
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(res[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(res);
//    delete[] x;
//    delete[] y;
//    delete[] res;

    return 0;
}
