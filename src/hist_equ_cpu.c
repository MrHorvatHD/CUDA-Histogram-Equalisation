#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define BINS 256

typedef struct _histogram
{
    unsigned int *R;
    unsigned int *G;
    unsigned int *B;
} histogram;

void histogramCPU(unsigned char *imageIn, histogram H, int width, int height, int cpp)
{

    // Each color channel is 1 byte long, there are 4 channels RED, BLUE, GREEN and ALPHA
    // The order is RED|GREEN|BLUE|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
    for (int i = 0; i < (height); i++)
        for (int j = 0; j < (width); j++)
        {
            H.R[imageIn[(i * width + j) * cpp]]++;
            H.G[imageIn[(i * width + j) * cpp + 1]]++;
            H.B[imageIn[(i * width + j) * cpp + 2]]++;
        }
}

//helper function to print results of histogram for debugging
void printHistogram(histogram H)
{
    printf("Colour\tNo. Pixels\n");
    for (int i = 0; i < BINS; i++)
    {
        if (H.R[i] > 0)
            printf("%dR\t%d\n", i, H.R[i]);
        if (H.G[i] > 0)
            printf("%dG\t%d\n", i, H.G[i]);
        if (H.B[i] > 0)
            printf("%dB\t%d\n", i, H.B[i]);
    }
}

void histCumCPU(histogram H)
{
    // calculate cumulative distances
    for (int i = 1; i < BINS; i++)
    {
        H.R[i] += H.R[i - 1];
        H.G[i] += H.G[i - 1];
        H.B[i] += H.B[i - 1];
    }
}

void imgEquCPU(unsigned char *imageIn, histogram H, int width, int height, int cpp)
{

    // find min non 0 element of each
    int c = 0;
    int r_min = H.R[c];
    int g_min = H.G[c];
    int b_min = H.B[c];

    while (r_min <= 0 && g_min <= 0 && b_min <= 0)
    {
        c++;

        r_min += H.R[c] * !r_min;
        g_min += H.G[c] * !g_min;
        b_min += H.B[c] * !b_min;
    }

    //printf("R: %d, G: %d, B: %d\n", r_min, g_min, b_min);

    // Calculate new color level values using the histogram equalization formula
    for (int i = 0; i < BINS; i++)
    {
        H.R[i] = (unsigned char)round((float)(H.R[i] - r_min) / (width * height - r_min) * (BINS - 1));
        H.G[i] = (unsigned char)round((float)(H.G[i] - g_min) / (width * height - g_min) * (BINS - 1));
        H.B[i] = (unsigned char)round((float)(H.B[i] - b_min) / (width * height - b_min) * (BINS - 1));
    }

    // write new levels to image
    for (int i = 0; i < (height); i++)
        for (int j = 0; j < (width); j++)
        {
            imageIn[(i * width + j) * cpp] = H.R[imageIn[(i * width + j) * cpp]];
            imageIn[(i * width + j) * cpp + 1] = H.G[imageIn[(i * width + j) * cpp + 1]];
            imageIn[(i * width + j) * cpp + 2] = H.B[imageIn[(i * width + j) * cpp + 2]];
        }

    //printf("%d %d %d\n", width, height, width * height);
}

int main(int argc, char **argv)
{

    // check if enough arguments
    if (argc < 2)
    {
        fprintf(stderr, "Not enough arguments\n");
        fprintf(stderr, "Usage: %s <IMAGE_PATH>\n", argv[0]);
        exit(1);
    }

    char *image_file = argv[1];

    // Initalize the histogram
    histogram H;
    H.B = (unsigned int *)calloc(BINS, sizeof(unsigned int));
    H.G = (unsigned int *)calloc(BINS, sizeof(unsigned int));
    H.R = (unsigned int *)calloc(BINS, sizeof(unsigned int));

    // load image into buffer array
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_file, &width, &height, &cpp, STBI_rgb);
    printf("%d %d %d\n", width, height, cpp);


    // if image doesn't exists
    if (!image_in){
        fprintf(stderr, "Error loading image %s!\n", image_file);
        return 1;
    }

    clock_t start, end;
    double elapsed;

    // start timing and run histogram equalisation
    start = clock();
    histogramCPU(image_in, H, width, height, cpp);
    histCumCPU(H);
    imgEquCPU(image_in, H, width, height, cpp);
    end = clock();

    // print elapsed time
    elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f ms\n", elapsed * 1000);

    // write back modified image
    stbi_write_jpg("equalised_img_cpu.jpg", width, height, cpp, image_in, 100);
    stbi_image_free(image_in);

    // print histogram for debugging
    //printHistogram(H);

    return 0;
}

// compile
// gcc hist_equ_cpu.c -o hist_equ_cpu -lm
// ./hist_equ_cpu <IMAGE_PATH>