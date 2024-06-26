#include <stdio.h>
#include <math.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define CSC(call)                                                                                       \
do {                                                                                                        \
        cudaError_t res = call;                                                                 \
        if (res != cudaSuccess) {                                                                   \
                fprintf(stderr, "ERROR in %s:%d. Message: %s\n",                        \
                                __FILE__, __LINE__, cudaGetErrorString(res));           \
                exit(0);                                                                                    \
        }                                                                                                       \
} while(0)


struct el {		// Тип элемента массива. Структура из двух полей
	double val;
};

struct comparator {
	__host__ __device__ bool operator()(el a, el b) {
		return abs(a.val) < abs(b.val);
	}
};

__global__ void take_col(el* dev, el* arr, int cur, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = gridDim.x * blockDim.x;
  while (idx < n) {
    if(idx < cur) {
      dev[idx].val = 0;
    } else {
      dev[idx] = arr[n * cur + idx];
    }
    idx += offset;
  }
}

__global__ void swap(el* arr, int cur, int id_mx, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = gridDim.x * blockDim.x;
  el tmp;
  while (idx < n + 1) {
    tmp = arr[cur + idx * n];
    arr[cur + idx * n] = arr[id_mx + idx * n];
    arr[id_mx + idx * n] = tmp;
    idx += offset;
  }
}

__global__ void change_rows(el* arr, int n, int cur) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int offsetx = blockDim.x * gridDim.x;
  int offsety = blockDim.y * gridDim.y;

  double coef;
  for (int x = cur + idx + 1; x < n; x += offsetx){
    coef = -arr[n * cur + x].val / arr[n * cur + cur].val;
    for (int y = cur + idy + 1; y < n + 1; y += offsety) {
       arr[n * y + x].val = arr[n * y + x].val + arr[n * y + cur].val * coef;
    }
  }
}

int main() {
	int n;
  comparator comp;
  FILE *fp = fopen("in.txt", "r");
  fscanf(fp, "%d", &n);
	el *arr = (el *)malloc(sizeof(el) * (n + 1) * n);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
		  fscanf(fp, "%lf", &arr[i + j * n].val);
	  }
	}
  for(int i = 0; i < n; i++) {
		  fscanf(fp, "%lf", &arr[i + n * n].val);
	}
  el *dev_arr;
  CSC(cudaMalloc(&dev_arr, sizeof(el) * n));
  el *arr1;
  CSC(cudaMalloc(&arr1, sizeof(el) * (n + 1) * n));
  CSC(cudaMemcpy(arr1, arr, sizeof(el) * (n + 1) * n, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  float time;
  CSC(cudaEventCreate(&start));
  CSC(cudaEventCreate(&stop));
  CSC(cudaEventRecord(start, 0));

  for(int i = 0; i < n - 1; i++) {
    take_col << < 512, 512 >> > (dev_arr, arr1, i, n);
    thrust::device_ptr<el> p_arr = thrust::device_pointer_cast(dev_arr);
	  thrust::device_ptr<el> res = thrust::max_element(p_arr, p_arr + n, comp);
    if((int)(res - p_arr) != i){
        swap << < 512, 512 >> > (arr1, i, (int)(res - p_arr), n);
    }
    change_rows << < dim3(16, 16), dim3(32, 32) >> > (arr1, n, i);
  }

  CSC(cudaEventRecord(stop, 0));
  CSC(cudaEventSynchronize(stop));
  CSC(cudaEventElapsedTime(&time, start, stop));
  fprintf(stderr, "time = %f\n", time);

  CSC(cudaMemcpy(arr, arr1, sizeof(el) * (n + 1) * n, cudaMemcpyDeviceToHost));
  double* ans = (double*)malloc(sizeof(double) * n);
  /*for(int i = 0; i < n; i++) {
		for(int j = 0; j < n + 1; j++) {
		  printf("%lf ", arr[i + j * n]);
	  }
    printf("\n");
	}*/
  double sum;
  for(int i = n - 1; i > -1; i--) {
    sum = 0;
		for(int j = i + 1; j < n; j++) {
		  sum = sum + arr[j * n + i].val * ans[j];
	  }
    ans[i] = (arr[n * n + i].val - sum) / arr[n * i + i].val;
	}
  /*for(int i = 0 ; i < n ; i++){
      printf("%0.10e ", ans[i]);
  }*/
  CSC(cudaGetLastError());
  cudaFree(dev_arr);
  cudaFree(arr1);
	free(arr);
	return 0;
}
