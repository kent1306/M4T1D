#include <iostream>
#include <cstdlib>
#include <time.h>
#include <chrono>

using namespace std::chrono;
using namespace std;

void randomArray(int array[], unsigned long size)
{
    for (unsigned long i = 0; i < size; i++)
    {
        array[i] = rand();
    }
}

int partition(int array[], int low, int high)
{
    int pivot = array[high];

    int i = low - 1;

    for (int j = low; j < high; j++)
    {
        if (array[j] <= pivot)
        {
            i++;

            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    int temp = array[i + 1];
    array[i + 1] = array[high];
    array[high] = temp;

    return i + 1;
}

void quickSort(int array[], int low, int high)
{
    if (low < high)
    {
        int pivotIndex = partition(array, low, high);

        quickSort(array, low, pivotIndex - 1);
        quickSort(array, pivotIndex + 1, high);
    }
}

bool checkSorted(int array[], unsigned long size)
{
    for (unsigned long i = 1; i < size; i++)
    {
        if (array[i - 1] > array[i])
        {
            return false;
        }
    }

    return true;
}

int main()
{
    unsigned long size;

    cout << "Enter array size: ";
    cin >> size;

    srand(42);

    int *array;

    array = (int *) malloc(size * sizeof(int));

    randomArray(array, size);

    auto start = high_resolution_clock::now();

    quickSort(array, 0, size - 1);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Array size: "
         << size << endl;

    cout << "Execution time: "
         << duration.count()
         << " microseconds" << endl;

    if (checkSorted(array, size))
    {
        cout << "Array sorted correctly." << endl;
    }
    else
    {
        cout << "Array sorting failed." << endl;
    }

    free(array);

    return 0;
}