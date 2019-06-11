#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "utils.h"

__device__ int binarySearch(int* arr, int val, int low, int high) {
	if (high <= low) 
        return (val > arr[low]) ? (low + 1) : low; 
  
    int mid = (low + high)/2; 
  
    //if(val == a[mid]) we dont support duplicates anyway
        //return mid+1; 
  
    if(val > arr[mid])
		return binarySearch(arr, val, mid+1, high);
		
    return binarySearch(arr, val, low, mid); // was mid-1
}

// Should use binary search!
__device__ int getIndex(int* subAux, int ownIndex, int nLow, int nTot) {
	//int numBeforeInOther = 0;
	//int indexInArr = 0;
	int scanIndex;
	int upperBound;
	bool partOfFirstArr = ownIndex < nLow;

	if(partOfFirstArr) {
		scanIndex = nLow; // Start scanning in 2nd arr
		upperBound = nTot;
	} 
	else {
		scanIndex = 0;
		upperBound = nLow;
	}

	//while (subAux[scanIndex] < subAux[ownIndex] && scanIndex < upperBound)
		//scanIndex++;

	//scanIndex++;

	//numBeforeInOther = partOfFirstArr ? scanIndex - nLow : scanIndex;

	scanIndex = binarySearch(subAux, subAux[ownIndex], scanIndex, upperBound-1);
	// printf("ScanIndex found: %d, by thread: %d, nTot: %d\n", scanIndex, ownIndex, nTot);
	// numBeforeInOther = partOfFirstArr ? indexInArr - nLow : indexInArr;

	return ownIndex + scanIndex - nLow;

	/*
	if(! partOfFirstArr)
		return ownIndex - nLow + numBeforeInOther;

	return ownIndex + numBeforeInOther;
	*/
}
// CANNOT HANDLE DUPLICATES
__global__ void mergeKernel(int* arr, int* aux, int low, int mid, int high) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int nLow = mid - low + 1; // optimize
	int nHigh = high - mid;
	int nTot = nLow + nHigh;
	
	if(idx >= nTot)
		return;

	int arrIndex = getIndex(&aux[low], idx, nLow, nTot);
	arr[low + arrIndex] = aux[low + idx];
	
	//printf("idx %d assigns %d to %d\n", idx, aux[low + idx], low + arrIndex);
}

// Just a sequential merge instead of nested kernel
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

	int nTot = high - low + 1; // number of threads to spawn
	if(nTot > 4096) { // Don't launch a kernel if the merge is small
		int numThreadsPerBlock = 256;
		int numBlocks = (nTot + numThreadsPerBlock - 1) / numThreadsPerBlock;
		//printf("ID %d: Spawning %d threads to merge %d elements\n", idx, numThreadsPerBlock*numBlocks, nTot);
		// merge(arr, aux, low, mid, high);
		mergeKernel<<<numBlocks, numThreadsPerBlock>>>(arr, aux, low, mid, high);
		//__syncthreads();
	} else {
		merge(arr, aux, low, mid, high);
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
		
		
		cudaMemcpy(auxArr, deviceArr, n * sizeof(int), cudaMemcpyDeviceToDevice); 
		mergeSort<<<numBlocks, numThreadsPerBlock>>>(deviceArr, auxArr, currentSize, n, width);
		
		// std::cout << "Num sorts: " << numSorts << ", num threads spawned: " << numThreadsPerBlock * numBlocks << std::endl;
		// usleep(1000000);
		// cudaMemcpy(arr, deviceArr, n * sizeof(int), cudaMemcpyDefault);
		// std::cout << "DevArr transferred to arr: ";
		// printArray(arr, n);
	}

	cudaMemcpy(arr, deviceArr, n * sizeof(int), cudaMemcpyDefault);

	cudaFree(deviceArr);
	cudaFree(auxArr);
}