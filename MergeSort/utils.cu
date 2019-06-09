#include <iostream>

using namespace std;

void printArray(int* arr, int length) {
	cout << "[ ";
	for (int i = 0; i < length; i++)
		cout << arr[i] << " ";

	cout << "]" << endl;
}

bool checkSorted(int* arr, int length) {
	for (int i = 1; i < length; i++)
		if (arr[i] < arr[i - 1])
			return false;

	return true;
}

void randomizeArray(int* arr, int length) {
	for (int i = 0; i < length; i++)
		arr[i] = rand() % length;
}

void swapPointers(int* a, int* b) {
	int* temp = a; // swap pointers
	a = b;
	b = temp;
}