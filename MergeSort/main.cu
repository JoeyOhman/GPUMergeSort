#include "sequential.h"
#include "parallel.h"
#include "utils.h"
#include <iostream>
#include <time.h>
//#include <chrono>

using namespace std;


double bench(int* arr, int length, bool gpu) {
	//chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
	startTimer();
	if(gpu) 
		mergeSortGPU(arr, length);
	else
		mergeSortSeq(arr, length);

	double duration = getTimeElapsed();
	//chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
	//long long duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000;
	return duration;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		cout << "Enter arguments: arrayLength runOnGPU (0 for sequential, 1 for GPU)" << endl;
		return 0;
	}
	srand (time(NULL));

	const int length = atoi(argv[1]);
	bool gpu = atoi(argv[2]);

	int* arr = new int[length];
	
	//cout << "Sorting.." << endl;

	for (int i = 0; i < 10; i++) {
		// cout << "Randomizing..." << endl;
		randomizeArray(arr, length);
		// cout << "Randomized!" << endl;
		
		// int* copyArr = getSortedCopy(arr, length);
		// printArray(arr, length);
		double duration = bench(arr, length, gpu);

		// printArray(arr, length);
		
		// cout << "Correct? " << boolalpha << arraysEqual(arr, copyArr, length) << endl;
		cout << "Correct? " << boolalpha << checkSorted(arr, length) << endl;

		cout << "Duration: " << duration << " seconds" << endl << endl;
	}

	return 0;
}
