#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CSC(call)                                                                                       \
do {                                                                                                        \
        cudaError_t res = call;                                                                 \
        if (res != cudaSuccess) {                                                                   \
                fprintf(stderr, "ERROR in %s:%d. Message: %s\n",                        \
                                __FILE__, __LINE__, cudaGetErrorString(res));           \
                exit(0);                                                                                    \
        }                                                                                                       \
} while(0)


__global__ void bleloch_scan(uint *arr1, uint *sums, int n){
        int idx = threadIdx.x;
        int offset = gridDim.x * blockDim.x;
        extern __shared__ uint sdata[];
        int elems = 0;
        uint last, tmp;
        while(elems < n){
                if(elems + blockIdx.x * blockDim.x + idx >= n){
                        break;
                }
                sdata[idx] = arr1[elems + blockIdx.x * blockDim.x + idx];
                __syncthreads();
                for (int i = 2; i <= blockDim.x; i = i * 2) {
                        if( (idx - (i / 2 - 1)) % i == 0){
                                sdata[idx + i / 2] = sdata[idx + i / 2] + sdata[idx];
                        }
                        __syncthreads();
                }
                if(idx == 0){
                        last = sdata[blockDim.x - 1];
                        sdata[blockDim.x - 1] = 0;
                }
                __syncthreads();
                for (int i = blockDim.x; i >= 2; i = i / 2) {
                        if( (idx - (i / 2 - 1)) % i == 0){
                                tmp = sdata[idx + i / 2];
                                sdata[idx + i / 2] = sdata[idx] + sdata[idx + i / 2];
                                sdata[idx] = tmp;
                        }
                        __syncthreads();
                }
                if(idx == 0){
                        sums[elems / blockDim.x + blockIdx.x] = last;
                }
                arr1[elems + blockDim.x * blockIdx.x + idx] = sdata[idx];
                elems += offset;
        }
}


__global__ void put_sums(uint *arr1, uint *sums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    while(idx < n) {
        arr1[idx] = arr1[idx] + sums[idx / blockDim.x];
        idx += offset;
    }
}

__host__ void main_scan(uint *arr1, int n, int num_of_blocks, int num_of_threads){
        int n1 = n / num_of_threads;
        int number_of_sums = n1;
        if(n1 % num_of_threads != 0){
                number_of_sums = number_of_sums - (number_of_sums % num_of_threads) + num_of_threads;
        }

        uint *sums;
        CSC(cudaMalloc(&sums, sizeof(uint) * number_of_sums));
        CSC(cudaMemset(sums, 0, number_of_sums * sizeof(uint)));

        bleloch_scan <<< num_of_blocks, num_of_threads, num_of_threads * sizeof(uint) >>> (arr1, sums, n);
        if(n1 > 1){
                main_scan (sums, number_of_sums, num_of_blocks, num_of_threads);
                put_sums <<< num_of_blocks, num_of_threads >>> (arr1, sums, n);
        }
        CSC(cudaGetLastError());
        cudaFree(sums);
}


__global__ void get_vals(uint *arr1, uint *vals, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    while(idx < n) {
        vals[idx] = ((arr1[idx] >> k) & 1);
        idx += offset;
    }
}

__global__ void rank_swap(uint *arr1, uint *pref, int n, int i, uint *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    int cur1, cur2;
    while(idx < n) {
        cur1 = (((arr1[idx]) >> i) & 1);
        cur2 = (((arr1[n - 1]) >> i) & 1);
        if(cur1 == 1){
                out[pref[idx] + (n - pref[n - 1] - cur2)] = arr1[idx];
        } else {
                out[idx - pref[idx]] = arr1[idx];
        }
        idx += offset;
    }
}

int main() {
        int n;
        int num_of_blocks = 512;
        int num_of_threads = 512;
        fread(&n,sizeof(int), 1, stdin);
        uint *arr = (uint*)malloc(sizeof(uint) * n);
        fread(arr,sizeof(uint), n, stdin);

        uint *arr1;
        CSC(cudaMalloc(&arr1, sizeof(uint) * n));
        CSC(cudaMemcpy(arr1, arr, sizeof(uint) * n, cudaMemcpyHostToDevice));

        int size = n;
        if(size % num_of_threads != 0){
                size = size - (size % num_of_threads) + num_of_threads;
        }

        uint *pref;
        CSC(cudaMalloc(&pref, sizeof(uint) * size));
        CSC(cudaMemset(pref, 0, size * sizeof(uint)));

        uint *out;
        CSC(cudaMalloc(&out, sizeof(uint) * n));

        for(int i = 0; i < 32; i++){
                get_vals <<< num_of_blocks, num_of_threads >>> (arr1, pref, n, i);
                main_scan(pref, size, num_of_blocks, num_of_threads);
                rank_swap <<< num_of_blocks, num_of_threads >>> (arr1, pref, n, i, out);
                CSC(cudaMemcpy(arr1, out, sizeof(uint) * n, cudaMemcpyDeviceToDevice));
        }
        CSC(cudaMemcpy(arr, arr1, sizeof(uint) * n, cudaMemcpyDeviceToHost));
        fwrite(arr, sizeof(uint), n, stdout);
        CSC(cudaGetLastError());
        cudaFree(arr1);
        cudaFree(pref);
        cudaFree(out);
        free(arr);
}
