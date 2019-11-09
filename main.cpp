#include <iostream>

__global__
void add(
        const float *const x,
        const float *const y,
        float *const res,
        const int n
)
{
    for (int i = 0; i < n; i++)
        res[i] = x[i] + y[i];
}

int main()
{
    int N = 1 << 20; // 1M elements

    float* x, *y, *res;

//    auto *x = new float[N];
//    auto *y = new float[N];
//    auto *res = new float[N];

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&res, N*sizeof(float));
    cudaDeviceSynchronize();

// initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


// Run kernel on 1M elements on the CPU

    add<<<1, 1>>>(N, x, y);
//    add(x, y, res, N);

// Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(res[i] - 3.0f));
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
