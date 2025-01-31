﻿﻿#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>

using namespace std;

#define CSC(call)                                                                                       \
do {                                                                                                        \
        cudaError_t res = call;                                                                 \
        if (res != cudaSuccess) {                                                                   \
                fprintf(stderr, "ERROR in %s:%d. Message: %s\n",                        \
                                __FILE__, __LINE__, cudaGetErrorString(res));           \
                exit(0);                                                                                    \
        }                                                                                                       \
} while(0)


typedef unsigned char uchar;

struct vec3 {
    double x;
    double y;
    double z;
};


__host__ __device__ double dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ vec3 prod(vec3 a, vec3 b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,  a.x * b.y - a.y * b.x };
}

__host__ __device__ vec3 norm(vec3 v) {
    double l = sqrt(dot(v, v));
    return { v.x / l, v.y / l, v.z / l };
}

__host__ __device__ vec3 diff(vec3 a, vec3 b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__host__ __device__ vec3 add(vec3 a, vec3 b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__host__ __device__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
    return { a.x * v.x + b.x * v.y + c.x * v.z,
                    a.y * v.x + b.y * v.y + c.y * v.z,
                    a.z * v.x + b.z * v.y + c.z * v.z };
}

__host__ __device__ vec3 to_normal(double r, double phi, double z) {
    return { r * cos(phi), r * sin(phi), z };
}

void print(vec3 v) {
    printf("%e %e %e\n", v.x, v.y, v.z);
}

vec3 dots3_1[60];

void make_points_icos(vec3 a, vec3 b, vec3 c, double d, int i) {
    vec3 norma = prod(diff(b, a), diff(c, a));
    norma = norm(norma);
    norma = { norma.x * d, norma.y * d, norma.z * d };

    dots3_1[i] = { a.x + norma.x, a.y + norma.y, a.z + norma.z };
    dots3_1[i + 1] = { b.x + norma.x, b.y + norma.y, b.z + norma.z };
    dots3_1[i + 2] = { c.x + norma.x, c.y + norma.y, c.z + norma.z };
}


struct trig {
    vec3 a;
    vec3 b;
    vec3 c;
    double4 color;
    double coef_refl;
    double coef_transp;
    bool is_edge = false;
    int pair = 0;
    int number_of_lights = 0;
    bool is_floor = false;
    double4 clr_floor = { 0.0, 0.0, 0.0, 255 };
};

trig matching_trig(vec3 a, vec3 b, vec3 c, double4 color, double coef_refl, double coef_transp, bool is_edge, int pair, int number_of_lights, bool is_floor, double4 clr_floor){
	trig A;
	A.a = a;
	A.b = b;
	A.c = c;
	A.color = color;
	A.coef_refl = coef_refl;
	A.coef_transp = coef_transp;
	A.is_edge = is_edge;
	A.pair = pair;
	A.number_of_lights = number_of_lights;
	A.is_floor = is_floor;
	A.clr_floor = clr_floor;
	return A;
};

struct hit {
    vec3 pos;
    vec3 normal;
    int num_of_trig;
};

struct ray {
    vec3 pos;
    vec3 dir;
    int id;
    double coef;
};

struct light {
    vec3 pos;
    double4 color;
};


trig* trigs;
int N = 207;

light* lights;
int light_numb;

uchar4* floor_text;
int w_f, h_f;

int build_space(vec3 c1, double4 clr1, double r1, double coef_refl1, double coef_transp1, int number_of_lights1,
    vec3 c2, double4 clr2, double r2, double coef_refl2, double coef_transp2, int number_of_lights2,
    vec3 c3, double4 clr3, double r3, double coef_refl3, double coef_transp3, int number_of_lights3,
    vec3 pnt1, vec3 pnt2, vec3 pnt3, vec3 pnt4, double4 clr_fl, double coef_fl, const char* floor_path) {

    //пол

    FILE* in = fopen(floor_path, "rb");

    if (in == NULL) {
        return 2;
    }

    fread(&w_f, sizeof(int), 1, in);
    fread(&h_f, sizeof(int), 1, in);
    floor_text = (uchar4*)malloc(sizeof(uchar4) * w_f * h_f);
    fread(floor_text, sizeof(uchar4), w_f * h_f, in);
    fclose(in);

    trigs[0] = matching_trig(pnt4, pnt3, pnt2, clr_fl, coef_fl, 0.0, false, 0, 0, true, clr_fl);
    trigs[1] = matching_trig(pnt1, pnt4, pnt2, clr_fl, coef_fl, 0.0, false, 0, 0, true, clr_fl);


    //грани фигур

    double4 clr4 = { 13.0 / 255.0, 42.0 / 255.0, 54.0 / 255.0, 255.0 };
    double coef_refl4 = 0.5;
    double coef_transp4 = 0.0;


    //гексаэдр

    double a1 = r1 / sqrt(3.0); // половина сторона куба
    double d = a1 / 10.0;     // ширина грани

    // точки гексаэдра
    vec3 dots1[24] = {
        //сдвиг по x
        {c1.x + a1, c1.y + a1 - d, c1.z + a1 - d},
        {c1.x + a1, c1.y + a1 - d, c1.z - a1 + d},
        {c1.x + a1, c1.y - a1 + d, c1.z + a1 - d},
        {c1.x + a1, c1.y - a1 + d, c1.z - a1 + d},

        {c1.x - a1, c1.y + a1 - d, c1.z + a1 - d},
        {c1.x - a1, c1.y + a1 - d, c1.z - a1 + d},
        {c1.x - a1, c1.y - a1 + d, c1.z + a1 - d},
        {c1.x - a1, c1.y - a1 + d, c1.z - a1 + d},

        //сдвиг по y

        {c1.x + a1 - d, c1.y + a1, c1.z + a1 - d},
        {c1.x + a1 - d, c1.y + a1, c1.z - a1 + d},
        {c1.x - a1 + d, c1.y + a1, c1.z + a1 - d},
        {c1.x - a1 + d, c1.y + a1, c1.z - a1 + d},

        {c1.x + a1 - d, c1.y - a1, c1.z + a1 - d},
        {c1.x + a1 - d, c1.y - a1, c1.z - a1 + d},
        {c1.x - a1 + d, c1.y - a1, c1.z + a1 - d},
        {c1.x - a1 + d, c1.y - a1, c1.z - a1 + d},

        //сдвиг по z

        {c1.x + a1 - d, c1.y + a1 - d, c1.z + a1},
        {c1.x + a1 - d, c1.y - a1 + d, c1.z + a1},
        {c1.x - a1 + d, c1.y + a1 - d, c1.z + a1},
        {c1.x - a1 + d, c1.y - a1 + d, c1.z + a1},

        {c1.x + a1 - d, c1.y + a1 - d, c1.z - a1},
        {c1.x + a1 - d, c1.y - a1 + d, c1.z - a1},
        {c1.x - a1 + d, c1.y + a1 - d, c1.z - a1},
        {c1.x - a1 + d, c1.y - a1 + d, c1.z - a1},

    };

    //грани

    trigs[2] = matching_trig( dots1[0], dots1[2], dots1[1],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);
    trigs[3] = matching_trig( dots1[3], dots1[1], dots1[2],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);

    trigs[4] = matching_trig( dots1[4], dots1[5], dots1[6],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);
    trigs[5] = matching_trig( dots1[7], dots1[6], dots1[5],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);

    trigs[6] = matching_trig( dots1[8], dots1[9], dots1[10],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);
    trigs[7] = matching_trig( dots1[11], dots1[10], dots1[9],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);

    trigs[8] = matching_trig( dots1[12], dots1[14], dots1[13],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);
    trigs[9] = matching_trig( dots1[15], dots1[13], dots1[14],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);

    trigs[10] = matching_trig( dots1[16], dots1[18], dots1[17],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);
    trigs[11] = matching_trig( dots1[19], dots1[17], dots1[18],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);

    trigs[12] = matching_trig( dots1[20], dots1[21], dots1[22],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);
    trigs[13] = matching_trig( dots1[23], dots1[22], dots1[21],  clr1, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);

    //точки

    trigs[14] = matching_trig( dots1[1], dots1[20], dots1[9],  clr4, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);
    trigs[15] = matching_trig( dots1[3], dots1[13], dots1[21],  clr4, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);

    trigs[16] = matching_trig( dots1[16], dots1[0], dots1[8],  clr4, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);
    trigs[17] = matching_trig( dots1[17], dots1[12], dots1[2],  clr4, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);

    trigs[18] = matching_trig( dots1[19], dots1[6], dots1[14],  clr4, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);
    trigs[19] = matching_trig( dots1[18], dots1[10], dots1[4],  clr4, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);

    trigs[20] = matching_trig( dots1[7], dots1[23], dots1[15],  clr4, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);
    trigs[21] = matching_trig( dots1[5], dots1[11], dots1[22],  clr4, coef_refl1, coef_transp1 , false, 0, 0, false, clr_fl);

    //рёбра

    d = d / 2;

    trigs[22] = matching_trig( dots1[0], dots1[1], dots1[8], clr4, coef_refl4, coef_transp4, true, 23, number_of_lights1 , false, clr_fl);
    trigs[23] = matching_trig( dots1[9], dots1[8], dots1[1], clr4, coef_refl4, coef_transp4, true, 22, number_of_lights1 , false, clr_fl);

    trigs[24] = matching_trig( dots1[2], dots1[0], dots1[17], clr4, coef_refl4, coef_transp4, true, 25, number_of_lights1 , false, clr_fl);
    trigs[25] = matching_trig( dots1[16], dots1[17], dots1[0], clr4, coef_refl4, coef_transp4, true, 24, number_of_lights1 , false, clr_fl);

    trigs[26] = matching_trig( dots1[1], dots1[3], dots1[20], clr4, coef_refl4, coef_transp4, true, 27, number_of_lights1 , false, clr_fl);
    trigs[27] = matching_trig( dots1[21], dots1[20], dots1[3], clr4, coef_refl4, coef_transp4, true, 26, number_of_lights1 , false, clr_fl);

    trigs[28] = matching_trig( dots1[12], dots1[13], dots1[2], clr4, coef_refl4, coef_transp4, true, 29, number_of_lights1 , false, clr_fl);
    trigs[29] = matching_trig( dots1[3], dots1[2], dots1[13], clr4, coef_refl4, coef_transp4, true, 28, number_of_lights1 , false, clr_fl);


    trigs[30] = matching_trig( dots1[10], dots1[11], dots1[4], clr4, coef_refl4, coef_transp4, true, 31, number_of_lights1 , false, clr_fl);
    trigs[31] = matching_trig( dots1[5], dots1[4], dots1[11], clr4, coef_refl4, coef_transp4, true, 30, number_of_lights1 , false, clr_fl);

    trigs[32] = matching_trig( dots1[4], dots1[6], dots1[18], clr4, coef_refl4, coef_transp4, true, 33, number_of_lights1 , false, clr_fl);
    trigs[33] = matching_trig( dots1[19], dots1[18], dots1[6], clr4, coef_refl4, coef_transp4, true, 32, number_of_lights1 , false, clr_fl);

    trigs[34] = matching_trig( dots1[22], dots1[23], dots1[5], clr4, coef_refl4, coef_transp4, true, 35, number_of_lights1 , false, clr_fl);
    trigs[35] = matching_trig( dots1[7], dots1[5], dots1[23], clr4, coef_refl4, coef_transp4, true, 34, number_of_lights1 , false, clr_fl);

    trigs[36] = matching_trig( dots1[6], dots1[7], dots1[14], clr4, coef_refl4, coef_transp4, true, 37, number_of_lights1 , false, clr_fl);
    trigs[37] = matching_trig( dots1[15], dots1[14], dots1[7], clr4, coef_refl4, coef_transp4, true, 36, number_of_lights1 , false, clr_fl);


    trigs[38] = matching_trig( dots1[20], dots1[22], dots1[9], clr4, coef_refl4, coef_transp4, true, 39, number_of_lights1 , false, clr_fl);
    trigs[39] = matching_trig( dots1[11], dots1[9], dots1[22], clr4, coef_refl4, coef_transp4, true, 38, number_of_lights1 , false, clr_fl);

    trigs[40] = matching_trig( dots1[8], dots1[10], dots1[16], clr4, coef_refl4, coef_transp4, true, 41, number_of_lights1 , false, clr_fl);
    trigs[41] = matching_trig( dots1[18], dots1[16], dots1[10], clr4, coef_refl4, coef_transp4, true, 40, number_of_lights1 , false, clr_fl);


    trigs[42] = matching_trig( dots1[13], dots1[15], dots1[21], clr4, coef_refl4, coef_transp4, true, 43, number_of_lights1 , false, clr_fl);
    trigs[43] = matching_trig( dots1[23], dots1[21], dots1[15], clr4, coef_refl4, coef_transp4, true, 42, number_of_lights1 , false, clr_fl);

    trigs[44] = matching_trig( dots1[14], dots1[12], dots1[19], clr4, coef_refl4, coef_transp4, true, 45, number_of_lights1 , false, clr_fl);
    trigs[45] = matching_trig( dots1[17], dots1[19], dots1[12], clr4, coef_refl4, coef_transp4, true, 44, number_of_lights1 , false, clr_fl);

    //октаэдр

    double a2 = r2 * sqrt(2.0) / 2.0;   // половина стороны октаэдра
    d = a2 / 20.0;

    // точки октаэдра
    vec3 dots2[24] = {
            {c2.x + d, c2.y, c2.z + r2 - d * sqrt(2)},
            {c2.x, c2.y + d, c2.z + r2 - d * sqrt(2)},
            {c2.x, c2.y - d, c2.z + r2 - d * sqrt(2)},
            {c2.x - d, c2.y, c2.z + r2 - d * sqrt(2)},

            {c2.x + d, c2.y, c2.z - r2 + d * sqrt(2)},
            {c2.x, c2.y + d, c2.z - r2 + d * sqrt(2)},
            {c2.x, c2.y - d, c2.z - r2 + d * sqrt(2)},
            {c2.x - d, c2.y, c2.z - r2 + d * sqrt(2)},

            {c2.x + a2 - d * sqrt(2), c2.y + a2, c2.z + d},
            {c2.x + a2 - d * sqrt(2), c2.y + a2, c2.z - d},
            {c2.x + a2, c2.y + a2 - d * sqrt(2), c2.z + d},
            {c2.x + a2, c2.y + a2 - d * sqrt(2), c2.z - d},

            {c2.x + a2 - d * sqrt(2), c2.y - a2, c2.z + d},
            {c2.x + a2 - d * sqrt(2), c2.y - a2, c2.z - d},
            {c2.x + a2, c2.y - a2 + d * sqrt(2), c2.z + d},
            {c2.x + a2, c2.y - a2 + d * sqrt(2), c2.z - d},

            {c2.x - a2 + d * sqrt(2), c2.y + a2, c2.z + d},
            {c2.x - a2 + d * sqrt(2), c2.y + a2, c2.z - d},
            {c2.x - a2, c2.y + a2 - d * sqrt(2), c2.z + d},
            {c2.x - a2, c2.y + a2 - d * sqrt(2), c2.z - d},

            {c2.x - a2 + d * sqrt(2), c2.y - a2, c2.z + d},
            {c2.x - a2 + d * sqrt(2), c2.y - a2, c2.z - d},
            {c2.x - a2, c2.y - a2 + d * sqrt(2), c2.z + d},
            {c2.x - a2, c2.y - a2 + d * sqrt(2), c2.z - d},
    };

    //вершины

    trigs[46] = matching_trig( dots2[0], dots2[1], dots2[2], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[47] = matching_trig( dots2[3], dots2[2], dots2[1], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[48] = matching_trig( dots2[4], dots2[6], dots2[5], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[49] = matching_trig( dots2[7], dots2[5], dots2[6], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[50] = matching_trig( dots2[8], dots2[10], dots2[9], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[51] = matching_trig( dots2[11], dots2[9], dots2[10], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[52] = matching_trig( dots2[14], dots2[12], dots2[15], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[53] = matching_trig( dots2[13], dots2[15], dots2[12], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[54] = matching_trig( dots2[16], dots2[17], dots2[18], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[55] = matching_trig( dots2[19], dots2[18], dots2[17], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[56] = matching_trig( dots2[20], dots2[23], dots2[21], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[57] = matching_trig( dots2[23], dots2[21], dots2[22], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    //рёбра

    trigs[58] = matching_trig( dots2[0], dots2[10], dots2[1], clr4, coef_refl4, coef_transp4, true, 59, number_of_lights2 , false, clr_fl);
    trigs[59] = matching_trig( dots2[8], dots2[1], dots2[10], clr4, coef_refl4, coef_transp4, true, 58, number_of_lights2 , false, clr_fl);

    trigs[60] = matching_trig( dots2[2], dots2[12], dots2[0], clr4, coef_refl4, coef_transp4, true, 61, number_of_lights2 , false, clr_fl);
    trigs[61] = matching_trig( dots2[14], dots2[0], dots2[12], clr4, coef_refl4, coef_transp4, true, 60, number_of_lights2 , false, clr_fl);

    trigs[62] = matching_trig( dots2[18], dots2[3], dots2[16], clr4, coef_refl4, coef_transp4, true, 63, number_of_lights2 , false, clr_fl);
    trigs[63] = matching_trig( dots2[1], dots2[16], dots2[3], clr4, coef_refl4, coef_transp4, true, 62, number_of_lights2 , false, clr_fl);

    trigs[64] = matching_trig( dots2[3], dots2[22], dots2[2], clr4, coef_refl4, coef_transp4, true, 65, number_of_lights2 , false, clr_fl);
    trigs[65] = matching_trig( dots2[20], dots2[2], dots2[22], clr4, coef_refl4, coef_transp4, true, 64, number_of_lights2 , false, clr_fl);


    trigs[66] = matching_trig( dots2[11], dots2[4], dots2[9], clr4, coef_refl4, coef_transp4, true, 67, number_of_lights2 , false, clr_fl);
    trigs[67] = matching_trig( dots2[5], dots2[9], dots2[4], clr4, coef_refl4, coef_transp4, true, 66, number_of_lights2 , false, clr_fl);

    trigs[68] = matching_trig( dots2[4], dots2[15], dots2[6], clr4, coef_refl4, coef_transp4, true, 69, number_of_lights2 , false, clr_fl);
    trigs[69] = matching_trig( dots2[13], dots2[6], dots2[15], clr4, coef_refl4, coef_transp4, true, 68, number_of_lights2 , false, clr_fl);

    trigs[70] = matching_trig( dots2[7], dots2[19], dots2[5], clr4, coef_refl4, coef_transp4, true, 71, number_of_lights2 , false, clr_fl);
    trigs[71] = matching_trig( dots2[17], dots2[5], dots2[19], clr4, coef_refl4, coef_transp4, true, 70, number_of_lights2 , false, clr_fl);

    trigs[72] = matching_trig( dots2[23], dots2[7], dots2[21], clr4, coef_refl4, coef_transp4, true, 73, number_of_lights2 , false, clr_fl);
    trigs[73] = matching_trig( dots2[6], dots2[21], dots2[7], clr4, coef_refl4, coef_transp4, true, 72, number_of_lights2 , false, clr_fl);


    trigs[74] = matching_trig( dots2[10], dots2[14], dots2[11], clr4, coef_refl4, coef_transp4, true, 75, number_of_lights2 , false, clr_fl);
    trigs[75] = matching_trig( dots2[15], dots2[11], dots2[14], clr4, coef_refl4, coef_transp4, true, 74, number_of_lights2 , false, clr_fl);

    trigs[76] = matching_trig( dots2[12], dots2[20], dots2[13], clr4, coef_refl4, coef_transp4, true, 77, number_of_lights2 , false, clr_fl);
    trigs[77] = matching_trig( dots2[21], dots2[13], dots2[20], clr4, coef_refl4, coef_transp4, true, 76, number_of_lights2 , false, clr_fl);

    trigs[78] = matching_trig( dots2[22], dots2[18], dots2[23], clr4, coef_refl4, coef_transp4, true, 79, number_of_lights2 , false, clr_fl);
    trigs[79] = matching_trig( dots2[19], dots2[23], dots2[18], clr4, coef_refl4, coef_transp4, true, 78, number_of_lights2 , false, clr_fl);

    trigs[80] = matching_trig( dots2[16], dots2[8], dots2[17], clr4, coef_refl4, coef_transp4, true, 81, number_of_lights2 , false, clr_fl);
    trigs[81] = matching_trig( dots2[9], dots2[17], dots2[8], clr4, coef_refl4, coef_transp4, true, 80, number_of_lights2 , false, clr_fl);

    //грани
    trigs[82] = matching_trig( dots2[0], dots2[14], dots2[10], clr2, coef_refl2, coef_transp2, false, 0, 0, false, clr_fl);
    trigs[83] = matching_trig( dots2[4], dots2[11], dots2[15], clr2, coef_refl2, coef_transp2, false, 0, 0, false, clr_fl);

    trigs[84] = matching_trig( dots2[1], dots2[8], dots2[16], clr2, coef_refl2, coef_transp2, false, 0, 0, false, clr_fl);
    trigs[85] = matching_trig( dots2[5], dots2[17], dots2[9], clr2, coef_refl2, coef_transp2 , false, 0, 0, false, clr_fl);

    trigs[86] = matching_trig( dots2[3], dots2[18], dots2[22], clr2, coef_refl2, coef_transp2 , false, 0, 0, false, clr_fl);
    trigs[87] = matching_trig( dots2[7], dots2[23], dots2[19], clr2, coef_refl2, coef_transp2 , false, 0, 0, false, clr_fl);

    trigs[88] = matching_trig( dots2[2], dots2[20], dots2[12], clr2, coef_refl2, coef_transp2 , false, 0, 0, false, clr_fl);
    trigs[89] = matching_trig( dots2[6], dots2[13], dots2[21], clr2, coef_refl2, coef_transp2 , false, 0, 0, false, clr_fl);


    //икосаэдр

    double phi = (1.0 + sqrt(5.0)) / 2.0; // золотое сечение

    // точки икосаэдра
    vec3 dots3[12] = {
            {phi, 1, 0},

            {phi, -1, 0},
            {1, 0, phi},
            {0, phi, 1},
            {0, phi, -1},
            {1, 0, -phi},


            {-phi, -1, 0},

            {0, -phi, 1},
            {-1, 0, phi},
            {-phi, 1, 0},
            {-1, 0, -phi},
            {0, -phi, -1}
    };

    for (int i = 0; i < 12; i++) {
        // продолжение формулы для координат икосаэдра с центром (0, 0, 0) и радиусом описанной окружности 1
        dots3[i].x = dots3[i].x / sqrt(1 + phi * phi);
        dots3[i].y = dots3[i].y / sqrt(1 + phi * phi);
        dots3[i].z = dots3[i].z / sqrt(1 + phi * phi);
        // приводим в соответствие для нашего радиуса
        dots3[i].x = dots3[i].x * r3;
        dots3[i].y = dots3[i].y * r3;
        dots3[i].z = dots3[i].z * r3;
        // смещаем в наш центр
        dots3[i].x = dots3[i].x + c3.x;
        dots3[i].y = dots3[i].y + c3.y;
        dots3[i].z = dots3[i].z + c3.z;
    }

    //грани

    d = r3 / 10.0;

    if (d > 1.0) {
        d = 1.0 / d;
    }

    // новые точки икосаэдра, учитывая ширину граней

    make_points_icos(dots3[1], dots3[0], dots3[2], d, 0);
    make_points_icos(dots3[0], dots3[3], dots3[2], d, 3);
    make_points_icos(dots3[2], dots3[3], dots3[8], d, 6);
    make_points_icos(dots3[8], dots3[9], dots3[6], d, 9);
    make_points_icos(dots3[6], dots3[10], dots3[11], d, 12);
    make_points_icos(dots3[11], dots3[5], dots3[1], d, 15);
    make_points_icos(dots3[1], dots3[5], dots3[0], d, 18);
    make_points_icos(dots3[8], dots3[3], dots3[9], d, 21);
    make_points_icos(dots3[6], dots3[9], dots3[10], d, 24);
    make_points_icos(dots3[11], dots3[10], dots3[5], d, 27);

    make_points_icos(dots3[7], dots3[1], dots3[2], d, 30);
    make_points_icos(dots3[7], dots3[2], dots3[8], d, 33);
    make_points_icos(dots3[7], dots3[8], dots3[6], d, 36);
    make_points_icos(dots3[7], dots3[6], dots3[11], d, 39);
    make_points_icos(dots3[7], dots3[11], dots3[1], d, 42);

    make_points_icos(dots3[4], dots3[0], dots3[5], d, 45);
    make_points_icos(dots3[4], dots3[5], dots3[10], d, 48);
    make_points_icos(dots3[4], dots3[10], dots3[9], d, 51);
    make_points_icos(dots3[4], dots3[9], dots3[3], d, 54);
    make_points_icos(dots3[4], dots3[3], dots3[0], d, 57);

    // грани

    for (int i = 0; i < 60;i = i + 3) {
        trigs[90 + i / 3] = matching_trig( dots3_1[i], dots3_1[i + 1], dots3_1[i + 2], clr3, coef_refl3, coef_transp3 , false, 0, 0, false, clr_fl);
    }

    // вершины

    trigs[111] = matching_trig( dots3_1[55], dots3_1[10], dots3_1[23], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[112] = matching_trig( dots3_1[55], dots3_1[25], dots3_1[10], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[113] = matching_trig( dots3_1[55], dots3_1[53], dots3_1[25], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[114] = matching_trig( dots3_1[24], dots3_1[38], dots3_1[11], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[115] = matching_trig( dots3_1[24], dots3_1[40], dots3_1[38], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[116] = matching_trig( dots3_1[25], dots3_1[12], dots3_1[40], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[117] = matching_trig( dots3_1[39], dots3_1[33], dots3_1[36], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[118] = matching_trig( dots3_1[39], dots3_1[30], dots3_1[33], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[119] = matching_trig( dots3_1[39], dots3_1[42], dots3_1[30], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[120] = matching_trig( dots3_1[32], dots3_1[6], dots3_1[34], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[121] = matching_trig( dots3_1[32], dots3_1[5], dots3_1[6], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[122] = matching_trig( dots3_1[32], dots3_1[2], dots3_1[5], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[123] = matching_trig( dots3_1[4], dots3_1[22], dots3_1[7], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[124] = matching_trig( dots3_1[4], dots3_1[56], dots3_1[22], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[125] = matching_trig( dots3_1[4], dots3_1[58], dots3_1[56], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[126] = matching_trig( dots3_1[8], dots3_1[37], dots3_1[35], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[127] = matching_trig( dots3_1[8], dots3_1[9], dots3_1[37], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[128] = matching_trig( dots3_1[8], dots3_1[21], dots3_1[9], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[129] = matching_trig( dots3_1[45], dots3_1[54], dots3_1[57], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[130] = matching_trig( dots3_1[45], dots3_1[51], dots3_1[54], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[131] = matching_trig( dots3_1[45], dots3_1[48], dots3_1[51], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[132] = matching_trig( dots3_1[27], dots3_1[41], dots3_1[14], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[133] = matching_trig( dots3_1[27], dots3_1[43], dots3_1[41], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[134] = matching_trig( dots3_1[27], dots3_1[15], dots3_1[43], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[135] = matching_trig( dots3_1[17], dots3_1[31], dots3_1[44], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[136] = matching_trig( dots3_1[17], dots3_1[0], dots3_1[31], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[137] = matching_trig( dots3_1[17], dots3_1[18], dots3_1[0], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[138] = matching_trig( dots3_1[20], dots3_1[3], dots3_1[1], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[139] = matching_trig( dots3_1[20], dots3_1[59], dots3_1[3], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[140] = matching_trig( dots3_1[20], dots3_1[46], dots3_1[59], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[141] = matching_trig( dots3_1[50], dots3_1[26], dots3_1[52], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[142] = matching_trig( dots3_1[50], dots3_1[13], dots3_1[26], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[143] = matching_trig( dots3_1[50], dots3_1[28], dots3_1[13], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    trigs[144] = matching_trig( dots3_1[29], dots3_1[19], dots3_1[16], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[145] = matching_trig( dots3_1[29], dots3_1[47], dots3_1[19], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);
    trigs[146] = matching_trig( dots3_1[19], dots3_1[49], dots3_1[47], clr4, coef_refl4, coef_transp4 , false, 0, 0, false, clr_fl);

    // рёбра

    trigs[147] = matching_trig( dots3_1[44], dots3_1[43], dots3_1[17], clr4, coef_refl4, coef_transp4, true, 148, number_of_lights3 , false, clr_fl);
    trigs[148] = matching_trig( dots3_1[15], dots3_1[17], dots3_1[43], clr4, coef_refl4, coef_transp4, true, 147, number_of_lights3 , false, clr_fl);

    trigs[149] = matching_trig( dots3_1[17], dots3_1[16], dots3_1[18], clr4, coef_refl4, coef_transp4, true, 150, number_of_lights3 , false, clr_fl);
    trigs[150] = matching_trig( dots3_1[19], dots3_1[18], dots3_1[16], clr4, coef_refl4, coef_transp4, true, 149, number_of_lights3 , false, clr_fl);

    trigs[151] = matching_trig( dots3_1[18], dots3_1[20], dots3_1[0], clr4, coef_refl4, coef_transp4, true, 152, number_of_lights3 , false, clr_fl);
    trigs[152] = matching_trig( dots3_1[1], dots3_1[0], dots3_1[20], clr4, coef_refl4, coef_transp4, true, 151, number_of_lights3 , false, clr_fl);

    trigs[153] = matching_trig( dots3_1[0], dots3_1[2], dots3_1[31], clr4, coef_refl4, coef_transp4, true, 154, number_of_lights3 , false, clr_fl);
    trigs[154] = matching_trig( dots3_1[32], dots3_1[31], dots3_1[2], clr4, coef_refl4, coef_transp4, true, 153, number_of_lights3 , false, clr_fl);

    trigs[155] = matching_trig( dots3_1[31], dots3_1[30], dots3_1[44], clr4, coef_refl4, coef_transp4, true, 156, number_of_lights3 , false, clr_fl);
    trigs[156] = matching_trig( dots3_1[42], dots3_1[44], dots3_1[30], clr4, coef_refl4, coef_transp4, true, 155, number_of_lights3 , false, clr_fl);


    trigs[157] = matching_trig( dots3_1[39], dots3_1[41], dots3_1[42], clr4, coef_refl4, coef_transp4, true, 158, number_of_lights3 , false, clr_fl);
    trigs[158] = matching_trig( dots3_1[43], dots3_1[42], dots3_1[41], clr4, coef_refl4, coef_transp4, true, 157, number_of_lights3 , false, clr_fl);

    trigs[159] = matching_trig( dots3_1[27], dots3_1[29], dots3_1[15], clr4, coef_refl4, coef_transp4, true, 160, number_of_lights3 , false, clr_fl);
    trigs[160] = matching_trig( dots3_1[16], dots3_1[15], dots3_1[29], clr4, coef_refl4, coef_transp4, true, 159, number_of_lights3 , false, clr_fl);

    trigs[161] = matching_trig( dots3_1[47], dots3_1[46], dots3_1[19], clr4, coef_refl4, coef_transp4, true, 162, number_of_lights3 , false, clr_fl);
    trigs[162] = matching_trig( dots3_1[20], dots3_1[19], dots3_1[46], clr4, coef_refl4, coef_transp4, true, 161, number_of_lights3 , false, clr_fl);

    trigs[163] = matching_trig( dots3_1[3], dots3_1[5], dots3_1[1], clr4, coef_refl4, coef_transp4, true, 164, number_of_lights3 , false, clr_fl);
    trigs[164] = matching_trig( dots3_1[2], dots3_1[1], dots3_1[5], clr4, coef_refl4, coef_transp4, true, 163, number_of_lights3 , false, clr_fl);

    trigs[165] = matching_trig( dots3_1[34], dots3_1[33], dots3_1[32], clr4, coef_refl4, coef_transp4, true, 166, number_of_lights3 , false, clr_fl);
    trigs[166] = matching_trig( dots3_1[30], dots3_1[32], dots3_1[33], clr4, coef_refl4, coef_transp4, true, 165, number_of_lights3 , false, clr_fl);


    trigs[167] = matching_trig( dots3_1[33], dots3_1[35], dots3_1[36], clr4, coef_refl4, coef_transp4, true, 168, number_of_lights3 , false, clr_fl);
    trigs[168] = matching_trig( dots3_1[37], dots3_1[36], dots3_1[35], clr4, coef_refl4, coef_transp4, true, 167, number_of_lights3 , false, clr_fl);

    trigs[169] = matching_trig( dots3_1[41], dots3_1[40], dots3_1[14], clr4, coef_refl4, coef_transp4, true, 170, number_of_lights3 , false, clr_fl);
    trigs[170] = matching_trig( dots3_1[12], dots3_1[14], dots3_1[40], clr4, coef_refl4, coef_transp4, true, 169, number_of_lights3 , false, clr_fl);

    trigs[171] = matching_trig( dots3_1[29], dots3_1[28], dots3_1[49], clr4, coef_refl4, coef_transp4, true, 172, number_of_lights3 , false, clr_fl);
    trigs[172] = matching_trig( dots3_1[50], dots3_1[49], dots3_1[28], clr4, coef_refl4, coef_transp4, true, 171, number_of_lights3 , false, clr_fl);

    trigs[173] = matching_trig( dots3_1[46], dots3_1[45], dots3_1[59], clr4, coef_refl4, coef_transp4, true, 174, number_of_lights3 , false, clr_fl);
    trigs[174] = matching_trig( dots3_1[57], dots3_1[59], dots3_1[45], clr4, coef_refl4, coef_transp4, true, 173, number_of_lights3 , false, clr_fl);

    trigs[175] = matching_trig( dots3_1[5], dots3_1[4], dots3_1[6], clr4, coef_refl4, coef_transp4, true, 176, number_of_lights3 , false, clr_fl);
    trigs[176] = matching_trig( dots3_1[7], dots3_1[6], dots3_1[4], clr4, coef_refl4, coef_transp4, true, 175, number_of_lights3 , false, clr_fl);


    trigs[177] = matching_trig( dots3_1[36], dots3_1[38], dots3_1[39], clr4, coef_refl4, coef_transp4, true, 178, number_of_lights3 , false, clr_fl);
    trigs[178] = matching_trig( dots3_1[40], dots3_1[39], dots3_1[38], clr4, coef_refl4, coef_transp4, true, 177, number_of_lights3 , false, clr_fl);

    trigs[179] = matching_trig( dots3_1[14], dots3_1[13], dots3_1[27], clr4, coef_refl4, coef_transp4, true, 180, number_of_lights3 , false, clr_fl);
    trigs[180] = matching_trig( dots3_1[28], dots3_1[27], dots3_1[13], clr4, coef_refl4, coef_transp4, true, 179, number_of_lights3 , false, clr_fl);

    trigs[181] = matching_trig( dots3_1[49], dots3_1[48], dots3_1[47], clr4, coef_refl4, coef_transp4, true, 182, number_of_lights3 , false, clr_fl);
    trigs[182] = matching_trig( dots3_1[45], dots3_1[47], dots3_1[48], clr4, coef_refl4, coef_transp4, true, 181, number_of_lights3 , false, clr_fl);

    trigs[183] = matching_trig( dots3_1[59], dots3_1[58], dots3_1[3], clr4, coef_refl4, coef_transp4, true, 184, number_of_lights3 , false, clr_fl);
    trigs[184] = matching_trig( dots3_1[4], dots3_1[3], dots3_1[58], clr4, coef_refl4, coef_transp4, true, 183, number_of_lights3 , false, clr_fl);

    trigs[185] = matching_trig( dots3_1[6], dots3_1[8], dots3_1[34], clr4, coef_refl4, coef_transp4, true, 186, number_of_lights3 , false, clr_fl);
    trigs[186] = matching_trig( dots3_1[35], dots3_1[34], dots3_1[8], clr4, coef_refl4, coef_transp4, true, 185, number_of_lights3 , false, clr_fl);


    trigs[187] = matching_trig( dots3_1[9], dots3_1[11], dots3_1[37], clr4, coef_refl4, coef_transp4, true, 188, number_of_lights3 , false, clr_fl);
    trigs[188] = matching_trig( dots3_1[38], dots3_1[37], dots3_1[11], clr4, coef_refl4, coef_transp4, true, 187, number_of_lights3 , false, clr_fl);

    trigs[189] = matching_trig( dots3_1[24], dots3_1[26], dots3_1[12], clr4, coef_refl4, coef_transp4, true, 190, number_of_lights3 , false, clr_fl);
    trigs[190] = matching_trig( dots3_1[13], dots3_1[12], dots3_1[26], clr4, coef_refl4, coef_transp4, true, 189, number_of_lights3 , false, clr_fl);

    trigs[191] = matching_trig( dots3_1[52], dots3_1[51], dots3_1[50], clr4, coef_refl4, coef_transp4, true, 192, number_of_lights3 , false, clr_fl);
    trigs[192] = matching_trig( dots3_1[48], dots3_1[50], dots3_1[51], clr4, coef_refl4, coef_transp4, true, 191, number_of_lights3 , false, clr_fl);

    trigs[193] = matching_trig( dots3_1[54], dots3_1[56], dots3_1[57], clr4, coef_refl4, coef_transp4, true, 194, number_of_lights3 , false, clr_fl);
    trigs[194] = matching_trig( dots3_1[58], dots3_1[57], dots3_1[56], clr4, coef_refl4, coef_transp4, true, 193, number_of_lights3 , false, clr_fl);

    trigs[195] = matching_trig( dots3_1[22], dots3_1[21], dots3_1[7], clr4, coef_refl4, coef_transp4, true, 196, number_of_lights3 , false, clr_fl);
    trigs[196] = matching_trig( dots3_1[8], dots3_1[7], dots3_1[21], clr4, coef_refl4, coef_transp4, true, 195, number_of_lights3 , false, clr_fl);


    trigs[197] = matching_trig( dots3_1[25], dots3_1[24], dots3_1[10], clr4, coef_refl4, coef_transp4, true, 198, number_of_lights3 , false, clr_fl);
    trigs[198] = matching_trig( dots3_1[11], dots3_1[10], dots3_1[24], clr4, coef_refl4, coef_transp4, true, 197, number_of_lights3 , false, clr_fl);

    trigs[199] = matching_trig( dots3_1[10], dots3_1[9], dots3_1[23], clr4, coef_refl4, coef_transp4, true, 200, number_of_lights3 , false, clr_fl);
    trigs[200] = matching_trig( dots3_1[21], dots3_1[23], dots3_1[9], clr4, coef_refl4, coef_transp4, true, 199, number_of_lights3 , false, clr_fl);

    trigs[201] = matching_trig( dots3_1[23], dots3_1[22], dots3_1[55], clr4, coef_refl4, coef_transp4, true, 202, number_of_lights3 , false, clr_fl);
    trigs[202] = matching_trig( dots3_1[56], dots3_1[55], dots3_1[22], clr4, coef_refl4, coef_transp4, true, 201, number_of_lights3 , false, clr_fl);

    trigs[203] = matching_trig( dots3_1[55], dots3_1[54], dots3_1[53], clr4, coef_refl4, coef_transp4, true, 204, number_of_lights3 , false, clr_fl);
    trigs[204] = matching_trig( dots3_1[51], dots3_1[53], dots3_1[54], clr4, coef_refl4, coef_transp4, true, 203, number_of_lights3 , false, clr_fl);

    trigs[205] = matching_trig( dots3_1[53], dots3_1[52], dots3_1[25], clr4, coef_refl4, coef_transp4, true, 206, number_of_lights3 , false, clr_fl);
    trigs[206] = matching_trig( dots3_1[26], dots3_1[25], dots3_1[52], clr4, coef_refl4, coef_transp4, true, 205, number_of_lights3 , false, clr_fl);

    return 0;

}

__host__ hit ray_cpu(vec3 pos, vec3 dir) {
    int k, k_min = -1;
    double ts_min = 0.0;
    for (k = 0; k < N; k++) {
        vec3 e1 = diff(trigs[k].b, trigs[k].a);
        vec3 e2 = diff(trigs[k].c, trigs[k].a);
        vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10)
            continue;
        vec3 t = diff(pos, trigs[k].a);
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0)
            continue;
        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;
        double ts = dot(q, e2) / div;
        if (ts < 0.0)
            continue;

        if (k_min == -1 || ts < ts_min)
        {
            k_min = k;
            ts_min = ts;
        }
    }

    if (k_min == -1)
        return { {0, 0, 0}, {0, 0, 0}, -1 };

    vec3 norma = prod(diff(trigs[k_min].b, trigs[k_min].a), diff(trigs[k_min].c, trigs[k_min].a));
    norma = norm(norma);
    vec3 place = add(pos, { dir.x * ts_min, dir.y * ts_min, dir.z * ts_min });
    return { place, norma, k_min };
}

__host__ int ray_trace_cpu(ray* rays, int size_of_rays, ray* rays_out, float4* data) {
    int ret_num = 0;

    for (int i = 0; i < size_of_rays; i++) {

        hit cur = ray_cpu(rays[i].pos, rays[i].dir);

        if (cur.num_of_trig == -1) {
            continue;
        }

        double4 res = { 0.0, 0.0, 0.0, 255 };

        double4 clr_d = {
               trigs[cur.num_of_trig].color.x * (1.0 - trigs[cur.num_of_trig].coef_transp),
               trigs[cur.num_of_trig].color.y * (1.0 - trigs[cur.num_of_trig].coef_transp),
               trigs[cur.num_of_trig].color.z * (1.0 - trigs[cur.num_of_trig].coef_transp),
               255
        };

        // обработка пола

        if (trigs[cur.num_of_trig].is_floor) {

            vec3 A = trigs[0].b;
            vec3 B = trigs[0].a;
            B = diff(B, A);
            vec3 C = trigs[0].c;
            C = diff(C, A);

            vec3 point = diff(cur.pos, A);

            double alpha = (point.x * B.y - point.y * B.x) / (C.x * B.y - C.y * B.x);
            double beta = (point.x * C.y - point.y * C.x) / (B.x * C.y - B.y * C.x);

            int xp = alpha * w_f;
            int yp = beta * h_f;
            xp = max(0, min(xp, w_f - 1));
            yp = max(0, min(yp, h_f - 1));


            clr_d = {
                (double)floor_text[yp * w_f + xp].x / 255.0 * trigs[cur.num_of_trig].clr_floor.x,
                (double)floor_text[yp * w_f + xp].y / 255.0 * trigs[cur.num_of_trig].clr_floor.y,
                (double)floor_text[yp * w_f + xp].z / 255.0 * trigs[cur.num_of_trig].clr_floor.z,
                255.0
            };
        }

        // обработка источников света на ребре

        bool is_light = false;

        if (trigs[cur.num_of_trig].is_edge && dot(cur.normal, rays[i].dir) > 0.0) {

            double radius =
                sqrt((trigs[cur.num_of_trig].b.x - trigs[trigs[cur.num_of_trig].pair].a.x) *
                    (trigs[cur.num_of_trig].b.x - trigs[trigs[cur.num_of_trig].pair].a.x) +
                    (trigs[cur.num_of_trig].b.y - trigs[trigs[cur.num_of_trig].pair].a.y) *
                    (trigs[cur.num_of_trig].b.y - trigs[trigs[cur.num_of_trig].pair].a.y) +
                    (trigs[cur.num_of_trig].b.z - trigs[trigs[cur.num_of_trig].pair].a.z) *
                    (trigs[cur.num_of_trig].b.z - trigs[trigs[cur.num_of_trig].pair].a.z)) / 2.0 * 0.7;

            vec3 point1 = add(trigs[cur.num_of_trig].b, trigs[trigs[cur.num_of_trig].pair].a);
            point1 = { point1.x / 2, point1.y / 2, point1.z / 2 };

            vec3 point2 = add(trigs[cur.num_of_trig].a, trigs[cur.num_of_trig].c);
            point2 = { point2.x / 2, point2.y / 2, point2.z / 2 };

            double shift_x = (point2.x - point1.x) / (trigs[cur.num_of_trig].number_of_lights + 1);
            double shift_y = (point2.y - point1.y) / (trigs[cur.num_of_trig].number_of_lights + 1);
            double shift_z = (point2.z - point1.z) / (trigs[cur.num_of_trig].number_of_lights + 1);

            vec3 cur_shift = point1;

            cur_shift.x = cur_shift.x + shift_x;
            cur_shift.y = cur_shift.y + shift_y;
            cur_shift.z = cur_shift.z + shift_z;

            double len;



            for (int k = 1; k <= trigs[cur.num_of_trig].number_of_lights; k++) {
                len = sqrt((cur_shift.x - cur.pos.x) * (cur_shift.x - cur.pos.x) +
                    (cur_shift.y - cur.pos.y) * (cur_shift.y - cur.pos.y) +
                    (cur_shift.z - cur.pos.z) * (cur_shift.z - cur.pos.z));
                if (len <= radius) {
                    is_light = true;
                    break;
                }
                cur_shift.x = cur_shift.x + shift_x;
                cur_shift.y = cur_shift.y + shift_y;
                cur_shift.z = cur_shift.z + shift_z;
            }
        }

	//коммент - костыль, без этих строчек получается хуже изображение, но нет артефактов
        /*if (dot(cur.normal, rays[i].dir) > 0) {
            cur.normal = diff({ 0, 0, 0 }, cur.normal);
        }*/

        // обработка источников света вокруг

        for (int k = 0;k < light_numb;k++) {

            hit tmp = ray_cpu(lights[k].pos, norm(diff(cur.pos, lights[k].pos)));

            if (cur.num_of_trig == tmp.num_of_trig) {

                //рассеянная составляющая света

                vec3 N_d = cur.normal;
                vec3 L_d = norm(diff(lights[k].pos, cur.pos));
                double d = sqrt(dot(diff(cur.pos, lights[k].pos), diff(cur.pos, lights[k].pos)));
                d = sqrt(d) / 2;

                res = {
                    res.x + clr_d.x * lights[k].color.x * dot(N_d, L_d) / d,
                    res.y + clr_d.y * lights[k].color.y * dot(N_d, L_d) / d,
                    res.z + clr_d.z * lights[k].color.z * dot(N_d, L_d) / d,
                    255
                };

                //зеркальная составляющая света

                vec3 L_s = norm(diff(cur.pos, lights[k].pos));
                vec3 R_s = {
                     L_s.x - 2 * dot(cur.normal, L_s) * cur.normal.x,
                     L_s.y - 2 * dot(cur.normal, L_s) * cur.normal.y,
                     L_s.z - 2 * dot(cur.normal, L_s) * cur.normal.z,
                };

                R_s = norm(R_s);
                vec3 S_s = norm(diff(rays[i].pos, cur.pos));
                double k_s = trigs[cur.num_of_trig].coef_refl;

                res = {
                    res.x + clr_d.x * k_s * lights[k].color.x * pow(dot(R_s, S_s), 9) / d,
                    res.y + clr_d.y * k_s * lights[k].color.y * pow(dot(R_s, S_s), 9) / d,
                    res.z + clr_d.z * k_s * lights[k].color.z * pow(dot(R_s, S_s), 9) / d,
                    255
                };
            }
        }

        // фоновая составляющая

        double4 ambient_light = { 0.3, 0.3, 0.3, 255 };

        res = {
              res.x + clr_d.x * ambient_light.x,
              res.y + clr_d.y * ambient_light.y,
              res.z + clr_d.z * ambient_light.z,
              255
        };

        if (is_light) {
            res = { 1.0, 1.0, 1.0, 255 };
        }

        //  учёт значения

        res = {
              res.x * rays[i].coef,
              res.y * rays[i].coef,
              res.z * rays[i].coef,
              255
        };

        data[rays[i].id].x = data[rays[i].id].x + res.x;
        data[rays[i].id].y = data[rays[i].id].y + res.y;
        data[rays[i].id].z = data[rays[i].id].z + res.z;

        vec3 shift;

        // отражённый луч

        if (trigs[cur.num_of_trig].coef_refl > 0) {

            vec3 R = {
                     rays[i].dir.x - 2 * dot(cur.normal, rays[i].dir) * cur.normal.x,
                     rays[i].dir.y - 2 * dot(cur.normal, rays[i].dir) * cur.normal.y,
                     rays[i].dir.z - 2 * dot(cur.normal, rays[i].dir) * cur.normal.z
            };
            R = norm(R);

            shift = {
                cur.pos.x + R.x * 0.001,
                cur.pos.y + R.y * 0.001,
                cur.pos.z + R.z * 0.001
            };

            rays_out[ret_num] = { shift, R, rays[i].id, rays[i].coef * trigs[cur.num_of_trig].coef_refl };
            ret_num++;
        }

        // луч, прошедший насквозь

        if (trigs[cur.num_of_trig].coef_transp > 0) {

            vec3 shift = {
                cur.pos.x + rays[i].dir.x * 0.001,
                cur.pos.y + rays[i].dir.y * 0.001,
                cur.pos.z + rays[i].dir.z * 0.001
            };

            rays_out[ret_num] = { shift, rays[i].dir, rays[i].id, rays[i].coef * trigs[cur.num_of_trig].coef_transp };
            ret_num++;
        }
    }

    return ret_num;
}

__host__ void init_rays_n_data_c(ray* rays, float4* data, vec3 pc, vec3 pv, int w, int h, double angle) {

    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, { 0.0, 0.0, 1.0 }));
    vec3 by = norm(prod(bx, bz));
    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {

            vec3 v = { -1.0 + dw * i, (-1.0 + dh * j) * h / w, z };
            vec3 dir = mult(bx, by, bz, v);
            dir = norm(dir);

            rays[i * h + j] = { pc, dir, (h - 1 - j) * w + i, 1.0 };
            data[i * h + j] = { 0.0, 0.0, 0.0, 255 };

        }

}

__host__ void write_data_c(uchar4* data2, float4* data1, int w, int h, int sqrt_ssaa) {

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            //ssaa
            double4 cur = { 0.0, 0.0, 0.0, 255.0 };
            for (int i1 = i * sqrt_ssaa; i1 < (i + 1) * sqrt_ssaa; i1++) {
                for (int j1 = j * sqrt_ssaa; j1 < (j + 1) * sqrt_ssaa; j1++) {
                    if (j1 * w * sqrt_ssaa + i1 >= w * h * sqrt_ssaa * sqrt_ssaa) {
                        break;
                    }
                    cur = {
                        cur.x + data1[j1 * w * sqrt_ssaa + i1].x,
                        cur.y + data1[j1 * w * sqrt_ssaa + i1].y,
                        cur.z + data1[j1 * w * sqrt_ssaa + i1].z,
                        255.0
                    };
                }
            }

            data2[j * w + i] = {
                (uchar)min(255.0 * cur.x / sqrt_ssaa / sqrt_ssaa, 255.0),
                (uchar)min(255.0 * cur.y / sqrt_ssaa / sqrt_ssaa, 255.0),
                (uchar)min(255.0 * cur.z / sqrt_ssaa / sqrt_ssaa, 255.0),
                255
            };

        }

}

__host__ int render_cpu(vec3 pc, vec3 pv, int w, int h, double angle, uchar4* data, int max_rec, int sqrt_ssaa) {

    int rays_numb = 0;
    
    w = w * sqrt_ssaa;
    h = h * sqrt_ssaa;

    float4* data1 = (float4*)malloc(sizeof(float4) * w * h);

    ray* rays = (ray*)malloc(sizeof(ray) * w * h);
    init_rays_n_data_c(rays, data1, pc, pv, w, h, angle);

    int size_of_rays = w * h;

    for (int i = 0;i < max_rec;i++) {
        if (size_of_rays < 1) {
            break;
        }
        rays_numb = rays_numb + size_of_rays;
        ray* rays_out = (ray*)malloc(sizeof(ray) * size_of_rays * 2);
        size_of_rays = ray_trace_cpu(rays, size_of_rays, rays_out, data1);
        free(rays);
        rays = rays_out;
    }

    w = w / sqrt_ssaa;
    h = h / sqrt_ssaa;

    write_data_c(data, data1, w, h, sqrt_ssaa);
    free(rays);
    free(data1);
    
    return rays_numb;
}

__device__ hit ray_gpu(trig* trigs, vec3 pos, vec3 dir, int N) {
    int k, k_min = -1;
    double ts_min = 0.0;
    for (k = 0; k < N; k++) {
        vec3 e1 = diff(trigs[k].b, trigs[k].a);
        vec3 e2 = diff(trigs[k].c, trigs[k].a);
        vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10)
            continue;
        vec3 t = diff(pos, trigs[k].a);
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0)
            continue;
        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;
        double ts = dot(q, e2) / div;
        if (ts < 0.0)
            continue;

        if (k_min == -1 || ts < ts_min)
        {
            k_min = k;
            ts_min = ts;
        }
    }

    if (k_min == -1)
        return { {0, 0, 0}, {0, 0, 0}, -1 };

    vec3 norma = prod(diff(trigs[k_min].b, trigs[k_min].a), diff(trigs[k_min].c, trigs[k_min].a));
    norma = norm(norma);
    vec3 place = add(pos, { dir.x * ts_min, dir.y * ts_min, dir.z * ts_min });
    return { place, norma, k_min };
}

__global__ void ray_trace_gpu(trig* trigs, int N, light* lights, int light_numb, ray* rays, int size_of_rays, ray* rays_out, int* size_of_rays_out, float4* data, uchar4* floor_text, int w_f, int h_f) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int i = idx; i < size_of_rays; i += offsetx) {

        hit cur = ray_gpu(trigs, rays[i].pos, rays[i].dir, N);

        if (cur.num_of_trig == -1) {
            continue;
        }

        double4 res = { 0.0, 0.0, 0.0, 255 };

        double4 clr_d = {
               trigs[cur.num_of_trig].color.x * (1.0 - trigs[cur.num_of_trig].coef_transp),
               trigs[cur.num_of_trig].color.y * (1.0 - trigs[cur.num_of_trig].coef_transp),
               trigs[cur.num_of_trig].color.z * (1.0 - trigs[cur.num_of_trig].coef_transp),
               255
        };

        // обработка пола

        if (trigs[cur.num_of_trig].is_floor) {

            vec3 A = trigs[0].b;
            vec3 B = trigs[0].a;
            B = diff(B, A);
            vec3 C = trigs[0].c;
            C = diff(C, A);

            vec3 point = diff(cur.pos, A);

            double alpha = (point.x * B.y - point.y * B.x) / (C.x * B.y - C.y * B.x);
            double beta = (point.x * C.y - point.y * C.x) / (B.x * C.y - B.y * C.x);

            int xp = alpha * w_f;
            int yp = beta * h_f;
            xp = max(0, min(xp, w_f - 1));
            yp = max(0, min(yp, h_f - 1));


            clr_d = {
                (double)floor_text[yp * w_f + xp].x / 255.0 * trigs[cur.num_of_trig].clr_floor.x,
                (double)floor_text[yp * w_f + xp].y / 255.0 * trigs[cur.num_of_trig].clr_floor.y,
                (double)floor_text[yp * w_f + xp].z / 255.0 * trigs[cur.num_of_trig].clr_floor.z,
                255.0
            };
        }

        // обработка источников света на ребре

        bool is_light = false;

        if (trigs[cur.num_of_trig].is_edge && dot(cur.normal, rays[i].dir) > 0.0) {

            double radius =
                sqrt((trigs[cur.num_of_trig].b.x - trigs[trigs[cur.num_of_trig].pair].a.x) *
                    (trigs[cur.num_of_trig].b.x - trigs[trigs[cur.num_of_trig].pair].a.x) +
                    (trigs[cur.num_of_trig].b.y - trigs[trigs[cur.num_of_trig].pair].a.y) *
                    (trigs[cur.num_of_trig].b.y - trigs[trigs[cur.num_of_trig].pair].a.y) +
                    (trigs[cur.num_of_trig].b.z - trigs[trigs[cur.num_of_trig].pair].a.z) *
                    (trigs[cur.num_of_trig].b.z - trigs[trigs[cur.num_of_trig].pair].a.z)) / 2.0 * 0.7;

            vec3 point1 = add(trigs[cur.num_of_trig].b, trigs[trigs[cur.num_of_trig].pair].a);
            point1 = { point1.x / 2, point1.y / 2, point1.z / 2 };

            vec3 point2 = add(trigs[cur.num_of_trig].a, trigs[cur.num_of_trig].c);
            point2 = { point2.x / 2, point2.y / 2, point2.z / 2 };

            double shift_x = (point2.x - point1.x) / (trigs[cur.num_of_trig].number_of_lights + 1);
            double shift_y = (point2.y - point1.y) / (trigs[cur.num_of_trig].number_of_lights + 1);
            double shift_z = (point2.z - point1.z) / (trigs[cur.num_of_trig].number_of_lights + 1);

            vec3 cur_shift = point1;

            cur_shift.x = cur_shift.x + shift_x;
            cur_shift.y = cur_shift.y + shift_y;
            cur_shift.z = cur_shift.z + shift_z;

            double len;

            for (int k = 1; k <= trigs[cur.num_of_trig].number_of_lights; k++) {
                len = sqrt((cur_shift.x - cur.pos.x) * (cur_shift.x - cur.pos.x) +
                    (cur_shift.y - cur.pos.y) * (cur_shift.y - cur.pos.y) +
                    (cur_shift.z - cur.pos.z) * (cur_shift.z - cur.pos.z));
                if (len <= radius) {
                    is_light = true;
                    break;
                }
                cur_shift.x = cur_shift.x + shift_x;
                cur_shift.y = cur_shift.y + shift_y;
                cur_shift.z = cur_shift.z + shift_z;
            }
        }

        if (dot(cur.normal, rays[i].dir) > 0) {
            cur.normal = diff({ 0, 0, 0 }, cur.normal);
        }

        // обработка источников света вокруг

        for (int k = 0;k < light_numb;k++) {

            hit tmp = ray_gpu(trigs, lights[k].pos, norm(diff(cur.pos, lights[k].pos)), N);

            if (cur.num_of_trig == tmp.num_of_trig) {

                //рассеянная составляющая света

                vec3 N_d = cur.normal;
                vec3 L_d = norm(diff(lights[k].pos, cur.pos));
                double d = sqrt(dot(diff(cur.pos, lights[k].pos), diff(cur.pos, lights[k].pos)));
                d = sqrt(d) / 2;

                res = {
                    res.x + clr_d.x * lights[k].color.x * dot(N_d, L_d) / d,
                    res.y + clr_d.y * lights[k].color.y * dot(N_d, L_d) / d,
                    res.z + clr_d.z * lights[k].color.z * dot(N_d, L_d) / d,
                    255
                };

                //зеркальная составляющая света

                vec3 L_s = norm(diff(cur.pos, lights[k].pos));
                vec3 R_s = {
                     L_s.x - 2 * dot(cur.normal, L_s) * cur.normal.x,
                     L_s.y - 2 * dot(cur.normal, L_s) * cur.normal.y,
                     L_s.z - 2 * dot(cur.normal, L_s) * cur.normal.z,
                };

                R_s = norm(R_s);
                vec3 S_s = norm(diff(rays[i].pos, cur.pos));
                double k_s = trigs[cur.num_of_trig].coef_refl;

                res = {
                    res.x + clr_d.x * k_s * lights[k].color.x * pow(dot(R_s, S_s), 9) / d,
                    res.y + clr_d.y * k_s * lights[k].color.y * pow(dot(R_s, S_s), 9) / d,
                    res.z + clr_d.z * k_s * lights[k].color.z * pow(dot(R_s, S_s), 9) / d,
                    255
                };
            }
        }

        // фоновая составляющая

        double4 ambient_light = { 0.3, 0.3, 0.3, 255 };

        res = {
              res.x + clr_d.x * ambient_light.x,
              res.y + clr_d.y * ambient_light.y,
              res.z + clr_d.z * ambient_light.z,
              255
        };

        if (is_light) {
            res = { 1.0, 1.0, 1.0, 255 };
        }

        //  учёт значения

        res = {
              res.x * rays[i].coef,
              res.y * rays[i].coef,
              res.z * rays[i].coef,
              255
        };

        atomicAdd(&data[rays[i].id].x, res.x);
        atomicAdd(&data[rays[i].id].y, res.y);
        atomicAdd(&data[rays[i].id].z, res.z);

        vec3 shift;

        // отражённый луч

        if (trigs[cur.num_of_trig].coef_refl > 0) {

            vec3 R = {
                     rays[i].dir.x - 2 * dot(cur.normal, rays[i].dir) * cur.normal.x,
                     rays[i].dir.y - 2 * dot(cur.normal, rays[i].dir) * cur.normal.y,
                     rays[i].dir.z - 2 * dot(cur.normal, rays[i].dir) * cur.normal.z
            };
            R = norm(R);

            shift = {
                cur.pos.x + R.x * 0.001,
                cur.pos.y + R.y * 0.001,
                cur.pos.z + R.z * 0.001
            };

            rays_out[atomicAdd(size_of_rays_out, 1)] = { shift, R, rays[i].id, rays[i].coef * trigs[cur.num_of_trig].coef_refl };
        }

        // луч, прошедший насквозь

        if (trigs[cur.num_of_trig].coef_transp > 0) {

            vec3 shift = {
                cur.pos.x + rays[i].dir.x * 0.001,
                cur.pos.y + rays[i].dir.y * 0.001,
                cur.pos.z + rays[i].dir.z * 0.001
            };

            rays_out[atomicAdd(size_of_rays_out, 1)] = { shift, rays[i].dir, rays[i].id, rays[i].coef * trigs[cur.num_of_trig].coef_transp };
        }
    }
}

__global__ void init_rays_n_data(ray* rays, float4* data, vec3 pc, vec3 pv, int w, int h, double angle) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, { 0.0, 0.0, 1.0 }));
    vec3 by = norm(prod(bx, bz));
    for (int i = idx; i < w; i += offsetx)
        for (int j = idy; j < h; j += offsety) {

            vec3 v = { -1.0 + dw * i, (-1.0 + dh * j) * h / w, z };
            vec3 dir = mult(bx, by, bz, v);
            dir = norm(dir);

            rays[i * h + j] = { pc, dir, (h - 1 - j) * w + i, 1.0 };
            data[i * h + j] = { 0.0, 0.0, 0.0, 255 };

        }

}

__global__ void write_data(uchar4* data2, float4* data1, int w, int h, int sqrt_ssaa) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < w; i += offsetx)
        for (int j = idy; j < h; j += offsety) {
            //ssaa
            double4 cur = { 0.0, 0.0, 0.0, 255.0 };
            for (int i1 = i * sqrt_ssaa; i1 < (i + 1) * sqrt_ssaa; i1++) {
                for (int j1 = j * sqrt_ssaa; j1 < (j + 1) * sqrt_ssaa; j1++) {
                    if (j1 * w * sqrt_ssaa + i1 >= w * h * sqrt_ssaa * sqrt_ssaa) {
                        break;
                    }
                    cur = {
                        cur.x + data1[j1 * w * sqrt_ssaa + i1].x,
                        cur.y + data1[j1 * w * sqrt_ssaa + i1].y,
                        cur.z + data1[j1 * w * sqrt_ssaa + i1].z,
                        255.0
                    };
                }
            }

            data2[j * w + i] = {
                (uchar)min(255.0 * cur.x / sqrt_ssaa / sqrt_ssaa, 255.0),
                (uchar)min(255.0 * cur.y / sqrt_ssaa / sqrt_ssaa, 255.0),
                (uchar)min(255.0 * cur.z / sqrt_ssaa / sqrt_ssaa, 255.0),
                255
            };

        }

}

int render_gpu(trig* trigs1, light* lights1, vec3 pc, vec3 pv, int w, int h, double angle, uchar4* data, int max_rec, int sqrt_ssaa, uchar4* floor_text1) {
    int rays_numb = 0;
    
    uchar4* data2;
    CSC(cudaMalloc(&data2, sizeof(uchar4) * w * h));

    w = w * sqrt_ssaa;
    h = h * sqrt_ssaa;

    float4* data1;
    CSC(cudaMalloc(&data1, sizeof(float4) * w * h));

    ray* rays;
    CSC(cudaMalloc(&rays, sizeof(ray) * w * h));
    init_rays_n_data << < dim3(16, 16), dim3(16, 16) >> > (rays, data1, pc, pv, w, h, angle);

    int size_of_rays = w * h;
    int* zero = (int*)malloc(sizeof(int));
    zero[0] = 0;

    for (int i = 0;i < max_rec;i++) {
        if (size_of_rays < 1) {
            break;
        }
        rays_numb = rays_numb + size_of_rays;
        ray* rays_out;
        CSC(cudaMalloc(&rays_out, sizeof(ray) * size_of_rays * 2));
        int* size_of_rays_out;
        CSC(cudaMalloc(&size_of_rays_out, sizeof(int)));
        CSC(cudaMemcpy(size_of_rays_out, zero, sizeof(int), cudaMemcpyHostToDevice));
        ray_trace_gpu << < 512, 512 >> > (trigs1, N, lights1, light_numb, rays, size_of_rays, rays_out, size_of_rays_out, data1, floor_text1, w_f, h_f);
        CSC(cudaFree(rays));
        rays = rays_out;
        cudaMemcpy(&size_of_rays, size_of_rays_out, sizeof(int), cudaMemcpyDeviceToHost);
        CSC(cudaFree(size_of_rays_out));
    }

    w = w / sqrt_ssaa;
    h = h / sqrt_ssaa;

    write_data << < dim3(16, 16), dim3(16, 16) >> > (data2, data1, w, h, sqrt_ssaa);
    cudaMemcpy(data, data2, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost);
    CSC(cudaFree(rays));
    CSC(cudaFree(data1));
    CSC(cudaFree(data2));
    
    return rays_numb;
}


int main(int argc, char* argv[]) {

    trigs = (trig*)malloc(sizeof(trig) * N);

    bool use_gpu;
    if(argc == 1){
    	use_gpu = true;
    } else if (argc != 2) {
        fprintf(stderr, "Неверное количество аргументов\n");
        return 1;
    } else if (strcmp(argv[1], "--gpu") == 0) {
        use_gpu = true;
    } else if (strcmp(argv[1], "--default") == 0) {
        printf("450\n");
	cout << "res/%03d.data \n";
	printf("1920 1080 120\n");

	printf("4.5 0.0 0.0 0.5 1.0 1.0 1.0 1.0 0.0 0.0\n");
	printf("2.0 0.0 0.0 1.0 0.1 1.0 1.0 1.0 0.0 0.0\n");

 	printf("-3.0  0.0  0.0  1.0  0.0  0.0  1.73   0.6   0.5  10\n");
 	printf(" 1.0  -2.0  0.0  0.0  1.0  0.0  1.41   0.2   0.5   5\n");
	printf(" 0.0  2.0   1.0  0.0  1.0  1.5  2.0    0.45  0.9   4\n");

	printf("-6.0 -6.0 -1.0   -6.0  6.0  -3.0   6.0 6.0 -1.0   6.0 -6.0 -3.0 in.data 0.75 0.75 0.75 0.5\n");

	printf("4\n");
	printf("-7  0 10     1.0 0.0 0.0\n");
	printf(" 0 -7 10     0.0 1.0 0.0\n");
	printf(" 0  7 10     0.0 0.0 1.0\n");
	printf(" 7  0 10     1.0 1.0 1.0\n");

	printf("10 2\n");
        return 0;
    } else if (strcmp(argv[1], "--cpu") != 0) {
        fprintf(stderr, "Передан неверный аргумент\n");
        return 2;
    }

    int k, w, h;

    string res_path1;
    double angle, r_0_c, z_0_c, phi_0_c, A_r_c, A_z_c, w_r_c, w_z_c, w_phi_c, p_r_c, p_z_c,
        r_0_n, z_0_n, phi_0_n, A_r_n, A_z_n, w_r_n, w_z_n, w_phi_n, p_r_n, p_z_n;

    vec3 c1, c2, c3;
    double4 clr1, clr2, clr3;
    double r1, r2, r3, coef_refl1, coef_refl2, coef_refl3, coef_transp1, coef_transp2, coef_transp3;
    int number_of_lights1, number_of_lights2, number_of_lights3;

    vec3 pnt1, pnt2, pnt3, pnt4;
    double4 clr_fl;
    double coef_fl;
    string floor_path1;

    int max_rec, sqrt_ssaa;

    cin >> k;

    cin >> res_path1;
    const char* res_path = res_path1.c_str();

    cin >> w >> h >> angle;

    cin >> r_0_c >> z_0_c >> phi_0_c;
    cin >> A_r_c >> A_z_c;
    cin >> w_r_c >> w_z_c >> w_phi_c;
    cin >> p_r_c >> p_z_c;

    cin >> r_0_n >> z_0_n >> phi_0_n;
    cin >> A_r_n >> A_z_n;
    cin >> w_r_n >> w_z_n >> w_phi_n;
    cin >> p_r_n >> p_z_n;

    cin >> c1.x >> c1.y >> c1.z;
    cin >> clr1.x >> clr1.y >> clr1.z;
    cin >> r1 >> coef_refl1 >> coef_transp1 >> number_of_lights1;

    cin >> c2.x >> c2.y >> c2.z;
    cin >> clr2.x >> clr2.y >> clr2.z;
    cin >> r2 >> coef_refl2 >> coef_transp2 >> number_of_lights2;

    cin >> c3.x >> c3.y >> c3.z;
    cin >> clr3.x >> clr3.y >> clr3.z;
    cin >> r3 >> coef_refl3 >> coef_transp3 >> number_of_lights3;

    cin >> pnt1.x >> pnt1.y >> pnt1.z;
    cin >> pnt2.x >> pnt2.y >> pnt2.z;
    cin >> pnt3.x >> pnt3.y >> pnt3.z;
    cin >> pnt4.x >> pnt4.y >> pnt4.z;

    cin >> floor_path1;
    const char* floor_path = floor_path1.c_str();

    cin >> clr_fl.x >> clr_fl.y >> clr_fl.z;
    cin >> coef_fl;

    cin >> light_numb;
    lights = (light*)malloc(sizeof(light) * light_numb);
    for (int i = 0; i < light_numb; i++) {
        cin >> lights[i].pos.x >> lights[i].pos.y >> lights[i].pos.z;
        cin >> lights[i].color.x >> lights[i].color.y >> lights[i].color.z;
    }

    cin >> max_rec;
    cin >> sqrt_ssaa;

    if (build_space(c1, clr1, r1, coef_refl1, coef_transp1, number_of_lights1,
        c2, clr2, r2, coef_refl2, coef_transp2, number_of_lights2,
        c3, clr3, r3, coef_refl3, coef_transp3, number_of_lights3,
        pnt1, pnt2, pnt3, pnt4, clr_fl, coef_fl, floor_path) == 2) {

        fprintf(stderr, "Что то не так с текстурой");
        return 2;
    }

    trig* trigs1;
    light* lights1;
    uchar4* floor_text1;
    if (use_gpu) {
        CSC(cudaMalloc(&trigs1, sizeof(trig) * N));
        CSC(cudaMemcpy(trigs1, trigs, sizeof(trig) * N, cudaMemcpyHostToDevice));

        CSC(cudaMalloc(&lights1, sizeof(light) * light_numb));
        CSC(cudaMemcpy(lights1, lights, sizeof(light) * light_numb, cudaMemcpyHostToDevice));

        CSC(cudaMalloc(&floor_text1, sizeof(uchar4) * w_f * h_f));
        CSC(cudaMemcpy(floor_text1, floor_text, sizeof(uchar4) * w_f * h_f, cudaMemcpyHostToDevice));
    }

    char buff[256];
    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    vec3 pc, pv;
    double t;

    for (int i = 0; i < k; i++) {

        t = 2 * M_PI * i / (k - 1);

        pc = to_normal(
            r_0_c + A_r_c * sin(w_r_c * t + p_r_c),
            phi_0_c + w_phi_c * t,
            z_0_c + A_z_c * sin(w_z_c * t + p_z_c)
        );

        pv = to_normal(
            r_0_n + A_r_n * sin(w_r_n * t + p_r_n),
            phi_0_n + w_phi_n * t,
            z_0_n + A_z_n * sin(w_z_n * t + p_z_n)
        );
        
    	cudaEvent_t start, stop;
    	CSC(cudaEventCreate(&start));
    	CSC(cudaEventCreate(&stop));
    	CSC(cudaEventRecord(start));
    	int rays_numb = 0;
        if (use_gpu) {
            rays_numb = render_gpu(trigs1, lights1, pc, pv, w, h, angle, data, max_rec, sqrt_ssaa, floor_text1);
        }
        else {
            rays_numb = render_cpu(pc, pv, w, h, angle, data, max_rec, sqrt_ssaa);
        }
        
        CSC(cudaDeviceSynchronize());
    	CSC(cudaGetLastError());

    	CSC(cudaEventRecord(stop));
    	CSC(cudaEventSynchronize(stop));
    	float cur_time;
    	CSC(cudaEventElapsedTime(&cur_time, start, stop));
    	CSC(cudaEventDestroy(start));
    	CSC(cudaEventDestroy(stop));

        sprintf(buff, res_path, i);
        printf("%d: %s\n", i, buff);
	printf("time (ms): %lf\n", cur_time);
	printf("rays: %d\n\n", rays_numb);
	
        FILE* out = fopen(buff, "wb");

        if (out == NULL) {
            fprintf(stderr, "Что-то не так с буфером вывода");
            return 2;
        }

        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(data, sizeof(uchar4), w * h, out);
        fclose(out);
    }
    if (use_gpu) {
        CSC(cudaFree(lights1));
        CSC(cudaFree(trigs1));
    }
    free(data);
    return 0;
    

}
