#include <stdio.h>
#include <stdlib.h>

int main(){
        int n = 10000001;
        unsigned int *arr = (unsigned int*)malloc(sizeof(unsigned int) * n);
        arr[0] = n - 1;
        for(int i = 1 ; i < n ;i++){
                arr[i] = rand();
        }
        FILE* fp = fopen("in.data", "wb");
        fwrite(arr, sizeof(unsigned int), n, fp);
}
