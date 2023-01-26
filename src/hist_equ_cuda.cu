#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// helper function to check if cuda executed correctly
#define checkCudaErrors(status)                            \
    if (status != cudaSuccess)                             \
    {                                                      \
        printf("Error: %s\n", cudaGetErrorString(status)); \
        exit(EXIT_FAILURE);                                \
    }

#define BLOCK_SIZE 1024 // 128 256 512 1024
#define HISTOGRAM_LEN 256
//----------------------------------------------------------------------
// KERNELS
//----------------------------------------------------------------------
__global__ void create_histogram(unsigned char *image, unsigned int *R, unsigned int *G, unsigned int *B, int width, int heigth, int cpp)
{
    // Privatized bins
    __shared__ unsigned int local_R[HISTOGRAM_LEN];
    __shared__ unsigned int local_G[HISTOGRAM_LEN];
    __shared__ unsigned int local_B[HISTOGRAM_LEN];

    for (unsigned int locIdx = threadIdx.x; locIdx < HISTOGRAM_LEN; locIdx += blockDim.x)
    {
        local_R[locIdx] = 0;
        local_G[locIdx] = 0;
        local_B[locIdx] = 0;
    }

    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;

    // All threads handle blockDim.x * gridDim.x
    // consecutive elements
    while (i < width * heigth)
    {
        atomicAdd(&local_R[image[i * cpp + 0]], 1);
        atomicAdd(&local_G[image[i * cpp + 1]], 1);
        atomicAdd(&local_B[image[i * cpp + 2]], 1);

        i += stride;
    }

    __syncthreads();

    for (unsigned int locIdx = threadIdx.x; locIdx < HISTOGRAM_LEN; locIdx += blockDim.x)
    {
        atomicAdd(&R[locIdx], local_R[locIdx]);
        atomicAdd(&G[locIdx], local_G[locIdx]);
        atomicAdd(&B[locIdx], local_B[locIdx]);
    }
}

__global__ void equalise_histogram(unsigned int *R, unsigned int *G, unsigned int *B, int width, int heigth)
{

    // uses 3 workgroups, one for each color
    // blockIdx.x defines color
    // threadIdx.x does the work for each color

    int idx = threadIdx.x;
    int offset;
    // printf("%d %d\n", blockIdx.x, idx);

    // perform work efficient Blelloch inclusive scan
    // down sweep with formula C[i + 2^k -1 + 2^k] += C[i + 2^k - 1]
    for (offset = 1; offset < HISTOGRAM_LEN; offset <<= 1)
    {

        // ensure that array is never out of bounds and we are accessing only the required bins
        if (idx + offset - 1 < HISTOGRAM_LEN && idx + 2 * offset - 1 < HISTOGRAM_LEN && idx % (2 * offset) == 0)
        {
            // printf("%d %d\n", idx + 2*offset -1, idx + offset - 1);

            // calculate R channel
            if (blockIdx.x == 0)
            {
                R[idx + 2 * offset - 1] += R[idx + offset - 1];
            }

            // calculate G channel
            else if (blockIdx.x == 1)
            {
                G[idx + 2 * offset - 1] += G[idx + offset - 1];
            }

            // calculate B channel
            else if (blockIdx.x == 2)
            {
                B[idx + 2 * offset - 1] += B[idx + offset - 1];
            }
        }

        __syncthreads();
    }

    // up sweep with formula C[i + 2^k -1 + 2^(k-1)] += C[i + 2^k - 1]
    for (offset /= 2; offset > 1; offset >>= 1)
    {

        // ensure that array is never out of bounds and we are accessing only the required bins
        if (idx + offset - 1 < HISTOGRAM_LEN && idx + offset + offset / 2 - 1 < HISTOGRAM_LEN && idx % offset == 0)
        {
            // printf("idx:%d %d %d\n",idx, idx + offset + offset/2 - 1, idx + offset - 1);

            // calculate R channel
            if (blockIdx.x == 0)
            {
                R[idx + offset + offset / 2 - 1] += R[idx + offset - 1];
            }

            // calculate G channel
            else if (blockIdx.x == 1)
            {
                G[idx + offset + offset / 2 - 1] += G[idx + offset - 1];
            }

            // calculate B channel
            else if (blockIdx.x == 2)
            {
                B[idx + offset + offset / 2 - 1] += B[idx + offset - 1];
            }
        }

        __syncthreads();
    }

    // we can not assume that the first element is non zero as needed for the equalisation formula
    // as each block works on different colors we can store that value in blocks shared memory
    __shared__ unsigned int minRGB;

    // one thread per block traverses to find first nonzero number
    if (idx == 0)
    {

        int c = 0;

        if (blockIdx.x == 0)
        {
            minRGB = R[0];

            while (minRGB <= 0)
            {
                c++;
                minRGB = R[c];
            }
        }

        else if (blockIdx.x == 1)
        {
            minRGB = G[0];

            while (minRGB <= 0)
            {
                c++;
                minRGB = G[c];
            }
        }

        else if (blockIdx.x == 2)
        {
            minRGB = B[0];

            while (minRGB <= 0)
            {
                c++;
                minRGB = B[c];
            }
        }
    }

    /*if(idx == 0)
        printf("B:%d %d\n", blockIdx.x, minRGB);*/

    // each block calculates new color levels for each color

    // normalize R channel
    if (blockIdx.x == 0)
        R[idx] = (unsigned char) round((float)(R[idx] - minRGB) / (width * heigth - minRGB) * (HISTOGRAM_LEN - 1));

    // normalize R channel
    else if (blockIdx.x == 1)
        G[idx] = (unsigned char) round((float)(G[idx] - minRGB) / (width * heigth - minRGB) * (HISTOGRAM_LEN - 1));

    // normalize B channel
    else if (blockIdx.x == 2)
        B[idx] = (unsigned char) round((float)(B[idx] - minRGB) / (width * heigth - minRGB) * (HISTOGRAM_LEN - 1));
}

__global__ void modify_image(unsigned char *image, unsigned int *R, unsigned int *G, unsigned int *B, int width, int heigth, int cpp)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;

    // All threads handle blockDim.x * gridDim.x
    // consecutive elements
    while (i < width * heigth)
    {
        image[i * cpp + 0] = R[image[i * cpp + 0]];
        image[i * cpp + 1] = G[image[i * cpp + 1]];
        image[i * cpp + 2] = B[image[i * cpp + 2]];

        i += stride;
    }
}

//----------------------------------------------------------------------
// SERIAL CODE
//----------------------------------------------------------------------

// helper function to print results of histogram for debugging
void printHistogram(unsigned int *R, unsigned int *G, unsigned int *B)
{
    printf("Colour\tNo. Pixels\n");
    for (int i = 0; i < HISTOGRAM_LEN; i++)
    {
        if (R[i] > 0)
            printf("%dR\t%d\n", i, R[i]);
        if (G[i] > 0)
            printf("%dG\t%d\n", i, G[i]);
        if (B[i] > 0)
            printf("%dB\t%d\n", i, B[i]);
    }
}

int main(int argc, char *argv[])
{

    // check if enough arguments
    if (argc < 2)
    {
        fprintf(stderr, "Not enough arguments\n");
        fprintf(stderr, "Usage: %s <IMAGE_PATH>\n", argv[0]);
        exit(1);
    }

    char *image_file = argv[1];

    // load image into buffer array
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_file, &width, &height, &cpp, STBI_rgb);
    printf("%d %d %d\n", width, height, cpp);

    // if image doesn't exists
    if (!image_in)
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
        return 1;
    }

    // allocate image on gpu
    int size = width * height * cpp;
    unsigned char *d_image;
    checkCudaErrors(cudaMalloc(&d_image, size));
    checkCudaErrors(cudaMemcpy(d_image, image_in, size, cudaMemcpyHostToDevice));

    // allocate histogram
    unsigned int *histR, *histG, *histB;
    checkCudaErrors(cudaMallocManaged(&histR, HISTOGRAM_LEN * sizeof(unsigned int)));
    checkCudaErrors(cudaMallocManaged(&histG, HISTOGRAM_LEN * sizeof(unsigned int)));
    checkCudaErrors(cudaMallocManaged(&histB, HISTOGRAM_LEN * sizeof(unsigned int)));

    // init histogram to zero
    for (int i = 0; i < HISTOGRAM_LEN; i++)
    {
        histR[i] = 0;
        histG[i] = 0;
        histB[i] = 0;
    }

    // prepare timing information
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // prepare and launch kernels (tailored to our graphics card)
    int blockSize = BLOCK_SIZE;
    int numBlocks = 40;
    create_histogram<<<numBlocks, blockSize>>>(d_image, histR, histG, histB, width, height, cpp);

    checkCudaErrors(cudaDeviceSynchronize());

    equalise_histogram<<<3, HISTOGRAM_LEN>>>(histR, histG, histB, width, height);

    checkCudaErrors(cudaDeviceSynchronize());

    modify_image<<<numBlocks, blockSize>>>(d_image, histR, histG, histB, width, height, cpp);

    checkCudaErrors(cudaDeviceSynchronize());

    // get modified image back
    checkCudaErrors(cudaMemcpy(image_in, d_image, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_image));

    // print results of kernel execution times
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f ms\n", elapsedTime);

    // write back modified image
    stbi_write_jpg("equalised_img_gpu.jpg", width, height, cpp, image_in, 100);
    stbi_image_free(image_in);

    // print histogram for debugging
    //printHistogram(histR, histG, histB);

    // free alocated memory
    cudaFree(histR);
    cudaFree(histG);
    cudaFree(histB);
    cudaFree(d_image);

    return 0;
}

// compile
// make SMS="75"
// ./hist_equ_cuda <IMAGE_PATH>