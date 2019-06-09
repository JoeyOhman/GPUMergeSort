#include "sequential.h"
#include "parallel.h"
#include "utils.h"
#include <iostream>
#include <chrono>

using namespace std;


long long bench(int* arr, int length, bool gpu) {
	chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

	if(gpu) 
		mergeSortGPU(arr, length);
	else
		mergeSortSeq(arr, length);

	chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
	long long duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000;
	return duration;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		cout << "Enter arguments: arrayLength runOnGPU (0 for sequential, 1 for GPU)" << endl;
		return 0;
	}
	const int length = atoi(argv[1]);
	bool gpu = atoi(argv[2]);

	int* arr = new int[length];
	
	//cout << "Sorting.." << endl;

	for (int i = 0; i < 1; i++) {
		randomizeArray(arr, length);
		printArray(arr, length);
		long long duration = bench(arr, length, gpu);

		printArray(arr, length);
		cout << "Sorted? " << boolalpha << checkSorted(arr, length) << endl;

		cout << "Duration: " << duration << " milliseconds" << endl << endl;
	}

	return 0;
}
