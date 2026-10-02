#include <iostream>
#include <cstdlib>
#include <time.h>
#include <chrono>
#include <omp.h>

using namespace std::chrono;
using namespace std;


// Temporary cutoff for OpenMP task creation.
// We will evaluate/tune this later.
const int TASK_CUTOFF = 10000;


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


void parallelQuickSort(int array[], int low, int high)
{
    if (low >= high)
    {
        return;
    }

    int size = high - low + 1;

    // Use sequential QuickSort for small partitions.
    if (size <= TASK_CUTOFF)
    {
        quickSort(array, low, high);
        return;
    }

    int pivotIndex = partition(array, low, high);

    #pragma omp task shared(array)
    {
        parallelQuickSort(array, low, pivotIndex - 1);
    }

    #pragma omp task shared(array)
    {
        parallelQuickSort(array, pivotIndex + 1, high);
    }

    #pragma omp taskwait
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


long long checksum(int array[], unsigned long size)
{
    long long sum = 0;

    for (unsigned long i = 0; i < size; i++)
    {
        sum += array[i];
    }

    return sum;
}


int main()
{
    unsigned long size;
    int numThreads;

    cout << "Enter array size: ";
    cin >> size;

    cout << "Enter number of threads: ";
    cin >> numThreads;


    int maxThreads = omp_get_max_threads();

    if (numThreads < 1 || numThreads > maxThreads)
    {
        cout << "Invalid number of threads." << endl;
        cout << "Maximum supported threads: "
             << maxThreads << endl;

        return 1;
    }


    srand(42);

    int *array;

    array = (int *) malloc(size * sizeof(int));

    randomArray(array, size);


    long long checksumBefore = checksum(array, size);


    omp_set_num_threads(numThreads);


    auto start = high_resolution_clock::now();


    #pragma omp parallel
    {
        #pragma omp single
        {
            parallelQuickSort(array, 0, size - 1);
        }
    }


    auto stop = high_resolution_clock::now();


    auto duration =
        duration_cast<microseconds>(stop - start);


    long long checksumAfter = checksum(array, size);


    cout << "Array size: "
         << size << endl;

    cout << "Number of threads: "
         << numThreads << endl;

    cout << "Task cutoff: "
         << TASK_CUTOFF << endl;

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


    if (checksumBefore == checksumAfter)
    {
        cout << "Checksum matched." << endl;
    }
    else
    {
        cout << "Checksum failed." << endl;
    }


    free(array);

    return 0;
}