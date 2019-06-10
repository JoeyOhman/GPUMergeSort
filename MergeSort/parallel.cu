#include <algorithm>
#include <iostream>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "utils.h"

// Should use binary search!
__device__ int getIndex(int* subAux, int ownIndex, int nLow, int nTot) {
	int indexArr;
	if(ownIndex >= nLow) // this thread is then part of 2nd arr
		indexArr = nLow;
	else
		indexArr = 0;

	while (subAux[indexArr] < subAux[ownIndex] && indexArr < nTot)
		indexArr++;

	return indexArr;
}
// CANNOT HANDLE DUPLICATES
__global__ void mergeKernel(int* arr, int* aux, int low, int mid, int high) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int nLow = mid - low + 1; // optimize
	int nHigh = high - mid;

	int arrIndex = getIndex(&aux[low], idx, nLow, nLow + nHigh);
	arr[low + arrIndex] = aux[idx];
}


__device__ void merge(int* arr, int* aux, int low, int mid, int high) {
	int i = 0;
	int j = 0;
	int mergedIndex = low;

	int nLow = mid - low + 1;
	int nHigh = high - mid;

	while (i < nLow && j < nHigh) {
		if (aux[low + i] <= aux[mid + 1 + j]) {
			arr[mergedIndex] = aux[low + i];
			i++;
		}
		else {
			arr[mergedIndex] = aux[mid + 1 + j];
			j++;
		}
		mergedIndex++;
	}

	while (i < nLow) {
		arr[mergedIndex] = aux[low + i];
		i++;
		mergedIndex++;
	}
	while (j < nHigh) {
		arr[mergedIndex] = aux[mid + 1 + j];
		j++;
		mergedIndex++;
	}
}

__global__ void mergeSort(int* arr, int* aux, int currentSize, int n, int width) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int low = idx * width;
	if(low >= n) return;
	int mid = low + currentSize - 1;
	int high = min(low + width - 1, n-1);

	// merge(arr, aux, low, mid, high);
	int nTot = high - low + 1; // number of threads to spawn
	int numThreadsPerBlock = 256;
	int numBlocks = (nTot + numThreadsPerBlock - 1) / numThreadsPerBlock;
	mergeKernel<<<numBlocks, numThreadsPerBlock>>>(arr, aux, low, mid, high);
}

void mergeFalloff(int* arr, int* aux, int low, int mid, int high) {
	int i = 0;
	int j = 0;
	int mergedIndex = low;

	int nLow = mid - low + 1;
	int nHigh = high - mid;

	while (i < nLow && j < nHigh) {
		if (aux[low + i] <= aux[mid + 1 + j]) {
			arr[mergedIndex] = aux[low + i];
			i++;
		}
		else {
			arr[mergedIndex] = aux[mid + 1 + j];
			j++;
		}
		mergedIndex++;
	}

	while (i < nLow) {
		arr[mergedIndex] = aux[low + i];
		i++;
		mergedIndex++;
	}
	while (j < nHigh) {
		arr[mergedIndex] = aux[mid + 1 + j];
		j++;
		mergedIndex++;
	}

}

void mergeSortFalloff(int* arr, int* aux, int currentSize, int n, int width) {
	for (int low = 0; low < n - currentSize; low += width) {
		int mid = low + currentSize - 1;
		int high = min(low + width - 1, n-1);

		mergeFalloff(arr, aux, low, mid, high);
	}
}

void mergeSortGPU(int* arr, int n) { // ASSUMES POWER OF 2 FOR NOW

	int* deviceArr;
	int* auxArr;

	cudaMallocManaged(&deviceArr, n * sizeof(int));
	cudaMallocManaged(&auxArr, n * sizeof(int)); // Allocate aux arr on GPU
	cudaMemcpy(deviceArr, arr, n * sizeof(int), cudaMemcpyDefault); // Move arr to cuda managed memory

	for (int currentSize = 1; currentSize < n; currentSize *= 2) {

		int width = currentSize*2;
		int numSorts = n / width; // number of sorting threads to spawn

		int numThreadsPerBlock = 256;
		int numBlocks = (numSorts + numThreadsPerBlock - 1) / numThreadsPerBlock;
		
		std::cout << "Num sorts: " << numSorts << ", num threads spawned: " << numThreadsPerBlock * numBlocks << std::endl;
		std::cout << "DevArr: ";
		printArray(deviceArr, n);

		/*
		if(numSorts >= 64) {
			cudaMemcpy(auxArr, deviceArr, n * sizeof(int), cudaMemcpyDeviceToDevice); 
			mergeSort<<<numBlocks, numThreadsPerBlock>>>(deviceArr, auxArr, currentSize, n, width);
		} else {
			cudaMemcpy(auxArr, deviceArr, n * sizeof(int), cudaMemcpyDefault);
			mergeSortFalloff(deviceArr, auxArr, currentSize, n, width);
		}
		*/
		cudaMemcpy(auxArr, deviceArr, n * sizeof(int), cudaMemcpyDeviceToDevice); 
		mergeSort<<<numBlocks, numThreadsPerBlock>>>(deviceArr, auxArr, currentSize, n, width);
	}

	cudaMemcpy(arr, deviceArr, n * sizeof(int), cudaMemcpyDefault);

	cudaFree(deviceArr);
	cudaFree(auxArr);
}