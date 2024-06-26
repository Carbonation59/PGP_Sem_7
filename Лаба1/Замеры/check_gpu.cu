%%cu
#include <stdio.h>

#define CSC(call)                                                       \
do {                                                                            \
        cudaError_t status = call;                              \
        if (status != cudaSuccess) {                                                                                                                                                            \
                fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));               \
                exit(0);                                                                \
        }                                                                                       \
} while(0)


__global__ void kernel(double *arr1, double *arr2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    while(idx < n) {
        arr1[idx] = arr1[idx] - arr2[idx];
        idx += offset;
    }
}

int main() {
    int n;
    FILE *in = fopen("test1_15.txt", "r");
    fscanf(in, "%d", &n);
    double *arr1 = (double *)malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++) {
        fscanf(in,"%lf",&arr1[i]);
    }

    double *arr2 = (double *)malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++) {
        fscanf(in, "%lf",&arr2[i]);
    }
    double *dev_arr1;
    cudaMalloc(&dev_arr1, sizeof(double) * n);
    cudaMemcpy(dev_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice);

    double *dev_arr2;
    cudaMalloc(&dev_arr2, sizeof(double) * n);
    cudaMemcpy(dev_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));

    kernel<<< 512, 512 >>>(dev_arr1, dev_arr2, n);

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    float t;
    CSC(cudaEventElapsedTime(&t, start, stop));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    cudaMemcpy(arr1, dev_arr1, sizeof(double) * n, cudaMemcpyDeviceToHost);
    /*for(int i = 0; i < n; i++){
        printf("%0.10e ", arr1[i]);
    }*/
    printf("%0.10e ", t);

    cudaFree(dev_arr1);
    cudaFree(dev_arr2);
    free(arr1);
    free(arr2);

    return 0;
}
