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


__device__ double get_bright(uchar4 p) {
        return (0.299 * double(p.x) + 0.587 * double(p.y) + 0.114 * double(p.z));
}

__device__ double minus(double p1, double p2) {
        return (p1 - p2);
}

__device__ int calc_grad(double Gx, double Gy) {
        int grad = round(pow((Gx * Gx + Gy * Gy), 0.5));
        return min(grad, 255);
}

__global__ void kernel(cudaTextureObject_t tex, uchar4* out, int w, int h) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        int offsetx = blockDim.x * gridDim.x;
        int offsety = blockDim.y * gridDim.y;

        int x, y;
        uchar4 p11, p12, p21, p22;
        double p11_bright, p12_bright, p21_bright, p22_bright;
        double Gx, Gy, Grad;
        for (y = idy; y < h; y += offsety)
                for (x = idx; x < w; x += offsetx) {
                        p11 = tex2D< uchar4 >(tex, x, y);
                        p11_bright = get_bright(p11);

                        p12 = tex2D< uchar4 >(tex, x + 1, y);
                        p12_bright = get_bright(p12);

                        p21 = tex2D< uchar4 >(tex, x, y + 1);
                        p21_bright = get_bright(p21);

                        p22 = tex2D< uchar4 >(tex, x + 1, y + 1);
                        p22_bright = get_bright(p22);

                        Gx = minus(p22_bright, p11_bright);
                        Gy = minus(p21_bright, p12_bright);
                        Grad = calc_grad(Gx, Gy);
                        out[y * w + x] = make_uchar4(Grad, Grad, Grad, p11.w);
                }
}

int main() {
        char name_in1[500];
        fgets(name_in1, 500, stdin);
        name_in1[strlen(name_in1) - 1] = 0;
        char* name_in = name_in1;

        char name_out1[500];
        fgets(name_out1, 500, stdin);
        name_out1[strlen(name_out1) - 1] = 0;
        char* name_out = name_out1;

        int w, h;
        FILE* fp = fopen(name_in, "rb");
        fread(&w, sizeof(int), 1, fp);
        fread(&h, sizeof(int), 1, fp);
        uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
        fread(data, sizeof(uchar4), w * h, fp);
        fclose(fp);

        cudaArray* arr;
        cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
        CSC(cudaMallocArray(&arr, &ch, w, h));

        CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = arr;

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;

        cudaTextureObject_t tex = 0;
        CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

        uchar4* dev_out;
        CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

        kernel << < dim3(16, 16), dim3(32, 32) >> > (tex, dev_out, w, h);
        CSC(cudaGetLastError());

        CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

        CSC(cudaDestroyTextureObject(tex));
        CSC(cudaFreeArray(arr));
        CSC(cudaFree(dev_out));

        fp = fopen(name_out, "wb");
        fwrite(&w, sizeof(int), 1, fp);
        fwrite(&h, sizeof(int), 1, fp);
        fwrite(data, sizeof(uchar4), w * h, fp);
        fclose(fp);

        free(data);
        return 0;
}
