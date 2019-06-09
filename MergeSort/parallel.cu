//#include <thrust/sort.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"


int* auxArr;

__global__ void merge(int* arrLow, int* arrHigh, int* auxLow, int* auxHigh, int nLow, int nHigh) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x; // number of threads
	int nTot = nLow + nHigh;
	// if (idx >= nLow + nHigh) return;

	for (int i = idx; i < nTot; i += stride) {
		auxLow[i] = arrLow[i];
	}

	
}

void mergeSort(int* arr, int low, int high) {
	if (low < high) {
		int mid = low + ((high - low) / 2);

		mergeSort(arr, low, mid);
		mergeSort(arr, mid + 1, high);

		int nLow = mid - low + 1;
		int nHigh = high - mid;
		int nTot = nLow + nHigh;
		int* auxArrLow = &auxArr[low];
		int* auxArrHigh = &auxArr[low + nLow];
		cudaMemcpy(auxArrLow, &arr[low], nTot, cudaMemcpyDeviceToDevice);
		
		int numThreadsPerBlock = 256;
		int numBlocks = nTot; // to be fixed
		//merge<<<numBlocks, numThreadsPerBlock >>> (&arr[low], &arr[mid + 1], auxLow, auxHigh, nLow, nHigh);
	}
}

void mergeSortGPU(int* arr, int length) {
	//thrust::sort(arr, arr + length);

	int *deviceArr;
	cudaMallocManaged(&deviceArr, length * sizeof(int));
	cudaMemcpy(deviceArr, arr, length, cudaMemcpyDefault); // Move arr to cuda managed memory
	cudaMemcpy(auxArr, deviceArr, length, cudaMemcpyDeviceToDevice); // Allocate aux arr on GPU
	mergeSort(arr, 0, length - 1);
}