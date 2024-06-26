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

__constant__ float arg1[33][3];
__constant__ float cov_rev1[33][3][3];

float arg[33][3];
float cov[33][3][3];
float cov_rev[33][3][3];

__host__ void put_zeroes(){
    for(int i = 0 ; i < 32 ; i++){
      arg[i][0] = 0;
      arg[i][1] = 0;
      arg[i][2] = 0;
  }

  for(int i = 0 ; i < 32 ; i++){
      for(int j = 0 ; j < 3 ; j++){
          for(int k = 0 ; k < 3 ; k++){
              cov[i][j][k] = 0;
              if(j == k){
                  cov_rev[i][j][k] = 1;
              } else {
                  cov_rev[i][j][k] = 0;
              }
          }
      }
  }
}

__host__ void get_arg_cov(uchar4 *data, int w, int h){
  int nc;
  int np;
  float cnt1, cnt2, cnt3;
	scanf("%d", &nc);
	for(int i = 0 ; i < nc; i++){
      scanf("%d", &np);
      int arr[np][2];
      for(int j = 0 ; j < np; j++){
          scanf("%d", &arr[j][0]);
          scanf("%d", &arr[j][1]);
          arg[i][0] = arg[i][0] + data[arr[j][0] + arr[j][1] * w].x;
          arg[i][1] = arg[i][1] + data[arr[j][0] + arr[j][1] * w].y;
          arg[i][2] = arg[i][2] + data[arr[j][0] + arr[j][1] * w].z;
      }
      arg[i][0] = arg[i][0] / np;
      arg[i][1] = arg[i][1] / np;
      arg[i][2] = arg[i][2] / np;

      for(int j = 0 ; j < np ; j++){
          cnt1 = data[arr[j][0] + arr[j][1] * w].x;
          cnt2 = data[arr[j][0] + arr[j][1] * w].y;
          cnt3 = data[arr[j][0] + arr[j][1] * w].z;

          cov[i][0][0] = (cov[i][0][0] + (cnt1 - arg[i][0]) * (cnt1 - arg[i][0]));
          cov[i][0][1] = (cov[i][0][1] + (cnt1 - arg[i][0]) * (cnt2 - arg[i][1]));
          cov[i][0][2] = (cov[i][0][2] + (cnt1 - arg[i][0]) * (cnt3 - arg[i][2]));

          cov[i][1][0] = (cov[i][1][0] + (cnt2 - arg[i][1]) * (cnt1 - arg[i][0]));
          cov[i][1][1] = (cov[i][1][1] + (cnt2 - arg[i][1]) * (cnt2 - arg[i][1]));
          cov[i][1][2] = (cov[i][1][2] + (cnt2 - arg[i][1]) * (cnt3 - arg[i][2]));

          cov[i][2][0] = (cov[i][2][0] + (cnt3 - arg[i][2]) * (cnt1 - arg[i][0]));
          cov[i][2][1] = (cov[i][2][1] + (cnt3 - arg[i][2]) * (cnt2 - arg[i][1]));
          cov[i][2][2] = (cov[i][2][2] + (cnt3 - arg[i][2]) * (cnt3 - arg[i][2]));
      }

      cov[i][0][0] = cov[i][0][0] / (np - 1);
      cov[i][0][1] = cov[i][0][1] / (np - 1);
      cov[i][0][2] = cov[i][0][2] / (np - 1);

      cov[i][1][0] = cov[i][1][0] / (np - 1);
      cov[i][1][1] = cov[i][1][1] / (np - 1);
      cov[i][1][2] = cov[i][1][2] / (np - 1);

      cov[i][2][0] = cov[i][2][0] / (np - 1);
      cov[i][2][1] = cov[i][2][1] / (np - 1);
      cov[i][2][2] = cov[i][2][2] / (np - 1);

  }
  arg[nc][0] = -1;
}

__host__ void reverse_cov(){
    float coef;
    for (int g = 0;g < 32;g++) {
      if(arg[g][0] == -1){
            break;
      }
      for (int i = 0;i < 3;i++) {
          coef = cov[g][i][i];
          for (int j = 0;j < 3;j++) {
              cov[g][i][j] = cov[g][i][j] / coef;
              cov_rev[g][i][j] = cov_rev[g][i][j] / coef;
          }
          for (int j = 0;j < 3;j++) {
              if (i == j) {
                  continue;
              }
              coef = cov[g][j][i];
              for (int k = 0;k < 3;k++) {
                  cov[g][j][k] = cov[g][j][k]  - coef * cov[g][i][k];
                  cov_rev[g][j][k] = cov_rev[g][j][k] - coef * cov_rev[g][i][k];
              }
          }
      }
    }
}

__device__ uchar4 mahalanobis(uchar4 p) {
    int id;
    float mx = -1e10;
    float cur[3];
    float cur1[3];
    float tmp;
    for(int i = 0 ; i < 33 ; i++){
        if(arg1[i][0] == -1){
            break;
        }
        cur[0] = float(p.x) - arg1[i][0];
	cur[1] = float(p.y) - arg1[i][1];
	cur[2] = float(p.z) - arg1[i][2];

        cur1[0] = 0;
	cur1[1] = 0;
	cur1[2] = 0;

        for(int j = 0;j < 3; j++){
            for(int k = 0;k < 3; k++){
                cur1[j] = cur1[j] - cov_rev1[i][k][j] * cur[k];
            }
        }
        tmp = cur[0] * cur1[0] + cur[1] * cur1[1] + cur[2] * cur1[2];
        if(tmp > mx){
            id = i;
            mx = tmp;
        }
    }
    p.w = id;
    return p;
}

__global__ void kernel(uchar4 *src, int w, int h) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  uchar4 cur;
  while(idx < w * h) {
        cur = src[idx];
        cur = mahalanobis(cur);
        src[idx] = cur;
        idx += offset;
  }
}

int main() {
  char name_in1[200];
  fgets(name_in1, 200, stdin);
  name_in1[strlen(name_in1) - 1] = 0;
  char* name_in = name_in1;

  char name_out1[200];
  fgets(name_out1, 200, stdin);
  name_out1[strlen(name_out1) - 1] = 0;
  char* name_out = name_out1;

  int w, h;
  FILE* fp = fopen(name_in, "rb");
  fread(&w, sizeof(int), 1, fp);
  fread(&h, sizeof(int), 1, fp);
  uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
  fread(data, sizeof(uchar4), w * h, fp);
  fclose(fp);

  put_zeroes();
  get_arg_cov(data, w, h);
  reverse_cov();

  uchar4 *data1;
	CSC(cudaMalloc(&data1, sizeof(uchar4) * w * h));
  cudaMemcpy(data1, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(arg1, arg, sizeof(float) * 33 * 3);
  cudaMemcpyToSymbol(cov_rev1, cov_rev, sizeof(float) * 33 * 3 * 3);

  kernel<<< 512, 512 >>>(data1, w, h);
  CSC(cudaGetLastError());

  CSC(cudaMemcpy(data, data1, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

  fp = fopen(name_out, "wb");
  fwrite(&w, sizeof(int), 1, fp);
  fwrite(&h, sizeof(int), 1, fp);
  fwrite(data, sizeof(uchar4), w * h, fp);
  fclose(fp);

  free(data);
  free(data1);
  return 0;
}
