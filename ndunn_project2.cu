#include <stdio.h>
#include <cuda.h>
#include <time.h>

/**
 * Nathan Dunn
 * CS-4370-90 Par. Prog. Many-Core GPUs
 * Professor Liu
 * 10-24-19
 * Tiled Matrix Multiplication
*/

#define N 16 // size of the matrices to be multiplied
#define TILE_WIDTH 2 // size of the tiles

/**
 * Computes the matrix multiplication on the CPU
 * m - First matrix to be multiplied	
 * n - Second matrix to be multiplied
 * p - Product of m and n
 * width - Size of the matrices being operated upon
*/
void MatrixMulOnHost(float *m, float *n, float *p, int width){
	for(int row = 0; row < width; ++row){
		for(int col = 0; col < width; ++col){
			double sum = 0;
			for(int k = 0; k < width; ++k){
				float a = m[row * width + k];
				float b = n[k * width + col];
				
				sum +=  a * b;
			}
			p[row * width + col] = sum;
		}
	}
}

/**
 * Computes the matrix multiplication on the GPU Device
 * d_M - First matrix to be multiplied	
 * d_N - Second matrix to be multiplied
 * p - Product of d_M and d_N
 * Width - Size of the matrices being operated upon
*/
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width)
{
	 __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	 __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

	 int bx = blockIdx.x;  int by = blockIdx.y;
	 int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	
	float Pvalue = 0;
	// Loop over the Md and Nd tiles required to compute the Pd element
	for (int m = 0; m < Width/TILE_WIDTH; ++m){
		
		// Collaborative loading of Md and Nd tiles into shared memory
		ds_M[ty][tx] = d_M[Row*Width + m*TILE_WIDTH+tx];
  		ds_N[ty][tx] = d_N[Col+(m*TILE_WIDTH+ty)*Width];
  		__syncthreads();
	   	for (int k = 0; k < TILE_WIDTH; ++k)
			 Pvalue += ds_M[ty][k] * ds_N[k][tx];
	 	__syncthreads();
	}	
	 d_P[Row*Width+Col] = Pvalue;
}

/**
	Verifies that an input matrix matches the product of two matrices. Each matrix
	element is computed individually and compared. If the comparison is not within
	the tolerance, the function automatically returns false
	A - Matrix to use for testing
	B - Matrix to use for testing
	C - Matrix to be tested
	width - size of input matrices
*/
bool verify(float *A, float *B, float *C, int  width) {
     const float relativeTolerance = 1e-6; // 1e-6 = 0.000001 
     for(int row = 0; row < width; ++row) {
    	for(int col = 0; col < width; ++col) {
     		float sum = 0;
      		for(unsigned int k = 0; k < width; ++k) {
        			sum += A[row*width + k]*B[k*width + col];
      		}
      		float relativeError = (sum - C[row*width + col])/sum;
     	 	if (relativeError > relativeTolerance
       		 || relativeError < -relativeTolerance) {
				 printf("TEST FAILED\n");
        			return false;
     	 	}
		}
	}
	printf("TEST PASSED\n");
    return true; 
}

/**
	Prints a matrix.
	matrix - matrix to be printed
	size - size of the matrix
*/
void printMatrix(float *matrix, int size){
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			printf("%f ", matrix[i * size + j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char* argv[]) 
{ 
	// matrices on the device
	float *a, *b, *c, *d;
	
	// matrices for the gpu
	float *dev_a, *dev_b, *dev_c;
	
	// allocate matrices
	a = (float*)malloc(sizeof(float) * N * N);
	b = (float*)malloc(sizeof(float) * N * N);
	c = (float*)malloc(sizeof(float) * N * N);
    d = (float*)malloc(sizeof(float) * N * N);
	
	// allocate device matrices
	cudaMalloc((void **)(&dev_a), N*N*sizeof(float));
	cudaMalloc((void **)(&dev_b), N*N*sizeof(float));
	cudaMalloc((void **)(&dev_c), N*N*sizeof(float));
	
	// initialize matrices a and b
	int init =1325;
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			int index = i * N + j;
			init=3125*init%65536;
			a[index]=(init-32768.0)/16384.0;
			init=3125*init%65536;
			b[index]=(init-32768.0)/16384.0;
		}
	}
	
	// Variables to measure GPU computation time and memory transfer time
	float timeGPU, timeTransfer, timeBack;
	cudaEvent_t gpuStart,gpuStop;
	
	// Begin measuring time for copying memory over to device
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuStop);
	cudaEventRecord(gpuStart,0);
	
	// copy array a,b (system memory) to dev_a, dev_b (device memory)
	cudaMemcpy(dev_a,a,N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,N*N*sizeof(float), cudaMemcpyHostToDevice);
	
	// Finish measuring time for copying memory over to device
	cudaDeviceSynchronize(); 
	cudaEventRecord(gpuStop,0);
	cudaEventSynchronize(gpuStop);
	cudaEventElapsedTime(&timeTransfer,gpuStart,gpuStop);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);
	
	printf("Matrix A: \n");
	printMatrix(a, N);
	printf("Matrix B: \n");
	printMatrix(b, N);
	
	// block and grid initialization for GPUs
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(N/dimBlock.x, N/dimBlock.y);
	
	// Begin measuring GPU computation time
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuStop);
	cudaEventRecord(gpuStart,0);
	
	// Launch kernels
	MatrixMulKernel<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
	
	// Finish measuring GPU computation time
	cudaDeviceSynchronize();
	cudaEventRecord(gpuStop,0);
	cudaEventSynchronize(gpuStop);
	cudaEventElapsedTime(&timeGPU,gpuStart,gpuStop);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);
	
	// Begin measuring time for copying memory back to host
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuStop);
	cudaEventRecord(gpuStart,0);
	
	// copy results from GPU back to system memory
	cudaMemcpy(c, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	
	// Finish measuring time for copying memory back to host
	cudaDeviceSynchronize();
    cudaEventRecord(gpuStop,0);
	cudaEventSynchronize(gpuStop);
	cudaEventElapsedTime(&timeBack,gpuStart,gpuStop);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);
	
	// total transfer time includes copy to and copy back time
    timeTransfer += timeBack;
	
	// display GPU device result
	printf("GPU Device Product: \n");
	printMatrix(c, N);
	
	// variables used to measure cpu computation time
	clock_t cpuStart, cpuEnd;
	float cpuTimeTaken;
	
	// start measuring cpu computation time
	cpuStart = clock();

	// compute result on CPU and display it
	MatrixMulOnHost(a, b, d, N);
	
	// stop measuring cpu computation time
	cpuEnd = clock();
	cpuTimeTaken = ((double)cpuEnd - cpuStart)/CLOCKS_PER_SEC; // in seconds 
	
	printf("CPU Product: \n");
	printMatrix(d, N);
	
	int cpuValid = verify(a, b, d, N);
	int gpuValid = verify(a, b, c, N);
	
	if(cpuValid && gpuValid){
		printf("Validating results...TEST PASSED\n");
	} else {
		printf("Validating results...TEST FAILED\n");
	}
	
	// display GPU computation and copy time
	printf("GPU Time: %f, Memory Copy Time: %f\n", timeGPU, timeTransfer);
	
	// display CPU computation time
	printf("CPU Time: %f\n", cpuTimeTaken);
	
	// free system and device memory
	free(a);
	free(b);
	free(c);
	free(d); 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
	return 0;
}