#include <stdio.h>
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC
#include <unistd.h> 

int main() {
    int n;
    FILE *in = fopen("test4_10000000.txt", "r");
    fscanf(in, "%d", &n);
    double *arr1 = (double *)malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++) {
        fscanf(in,"%lf",&arr1[i]);
    }

    double *arr2 = (double *)malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++) {
        fscanf(in, "%lf",&arr2[i]);
    }

    double time_spent = 0.0;

    clock_t begin = clock();

    for(int i = 0; i < n; i++) {
        arr1[i] = arr1[i] - arr2[i];
    }

    clock_t end = clock();

    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

    printf("%0.10e ", time_spent);

    return 0;
}
