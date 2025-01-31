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

__host__ void take_col(el* dev, el* arr, int cur, int n) {
  int idx = 0;
  while (idx < n) {
    if(idx < cur) {
      dev[idx].val = 0;
    } else {
      dev[idx] = arr[n * cur + idx];
    }
    idx += 1;
  }
}

__host__ void swap(el* arr, int cur, int id_mx, int n) {
  int idx = 0;
  int offset = 1;
  el tmp;
  while (idx < n + 1) {
    tmp = arr[cur + idx * n];
    arr[cur + idx * n] = arr[id_mx + idx * n];
    arr[id_mx + idx * n] = tmp;
    idx += offset;
  }
}

__host__ void change_rows(el* arr, int n, int cur) {
  int idx = 0;
  int idy = 0;
  int offsetx = 1;
  int offsety = 1;

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
  el *dev_arr = (el *)malloc(sizeof(el) * n);

  cudaEvent_t start, stop;
  float time;
  CSC(cudaEventCreate(&start));
  CSC(cudaEventCreate(&stop));
  CSC(cudaEventRecord(start, 0));

  for(int i = 0; i < n - 1; i++) {
    take_col(dev_arr, arr, i, n);
    if((int)(n - i) != i){
        swap(arr, i, (int)(n - i), n);
    }
    change_rows(arr, n, i);
  }

  CSC(cudaEventRecord(stop, 0));
  CSC(cudaEventSynchronize(stop));
  CSC(cudaEventElapsedTime(&time, start, stop));
  fprintf(stderr, "time = %f\n", time);

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
 // CSC(cudaGetLastError());
 // cudaFree(dev_arr);
 // cudaFree(arr1);
	free(arr);
	return 0;
}
