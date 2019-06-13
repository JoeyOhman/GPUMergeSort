#include "sequential.h"
#include "parallel.h"
#include "utils.h"
#include <iostream>
#include <stdio.h>
#include <time.h>
//#include <chrono>

using namespace std;


double bench(int* arr, int length, bool gpu) {
	startTimer();
	if(gpu) 
		mergeSortGPU(arr, length);
	else
		mergeSortSeq(arr, length);

	double duration = getTimeElapsed();
	return duration;
}


int main(int argc, char** argv) {
	if (argc < 3) {
		cout << "Enter arguments: arrayLength runOnGPU (0 for sequential, 1 for GPU)" << endl;
		return 0;
	}
	srand (time(NULL));
	double lowestTime = 9999999999.0;

	const int length = atoi(argv[1]);
	bool gpu = atoi(argv[2]);

	int* arr = new int[length];
	
	//cout << "Sorting.." << endl;

	for (int i = 0; i < 20; i++) {
		randomizeArray(arr, length);
		
		// printArray(arr, length);
		double duration = bench(arr, length, gpu);
		if(duration < lowestTime)
			lowestTime = duration;

		// printArray(arr, length);
		bool correct = checkSorted(arr, length);
		if(! correct) {
			cout << "INCORRECT SORT!" << endl;
			return 0;
		}
		//cout << "Correct? " << boolalpha << correct << endl;

		cout << "Duration: " << duration << " seconds" << endl;
	}
	cout << "Lowest time: " << lowestTime << " seconds" << endl;

	return 0;
}
