#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <stdio.h>
#include <omp.h>

using namespace std::chrono;
using namespace std;

// Adaptive configuration
const unsigned long THRESHOLD_1 = 35000;
const unsigned long THRESHOLD_2 = 4000000;

const int OPENMP_THREADS = 12;
const int TASK_CUTOFF = 10000;

const size_t GLOBAL_WORK_SIZE = 4096;
const size_t LOCAL_WORK_SIZE = 64;

// OpenCL variables
cl_platform_id platform;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_mem bufArray;

int err;

// Functions
void randomArray(
    int array[],
    unsigned long size
);

int partition(
    int array[],
    int low,
    int high
);

void quickSort(
    int array[],
    int low,
    int high
);

void parallelQuickSort(
    int array[],
    int low,
    int high
);

bool checkSorted(
    int array[],
    unsigned long size
);

long long calculateChecksum(
    int array[],
    unsigned long size
);

void mergeSortedChunksOpenMP(
    int array[],
    unsigned long size,
    unsigned long chunkSize
);

cl_device_id create_device();

void setup_openCL_device_context_queue_kernel(
    char *filename,
    char *kernelname
);

cl_program build_program(
    cl_context ctx,
    cl_device_id dev,
    const char *filename
);

void printDeviceInfo();

void free_opencl_memory();

int main()
{
    unsigned long size;

    cout << "Enter array size: ";
    cin >> size;

    srand(42);

    int *array =
        (int *) malloc(
            size * sizeof(int)
        );

    if (array == NULL)
    {
        cout << "Memory allocation failed."
             << endl;

        return 1;
    }

    randomArray(
        array,
        size
    );

    long long checksumBefore =
        calculateChecksum(
            array,
            size
        );

    cout << endl;

    cout << "Array size: "
         << size
         << endl;

    cout << "Threshold 1: "
         << THRESHOLD_1
         << endl;

    cout << "Threshold 2: "
         << THRESHOLD_2
         << endl;

    // SEQUENTIAL
    if (size < THRESHOLD_1)
    {
        cout << "Selected method: "
             << "Sequential QuickSort"
             << endl;

        auto start =
            high_resolution_clock::now();

        quickSort(
            array,
            0,
            size - 1
        );

        auto stop =
            high_resolution_clock::now();


        auto duration =
            duration_cast<microseconds>(
                stop - start
            );

        cout << "Execution time: "
             << duration.count()
             << " microseconds"
             << endl;
    }

    // OPENMP
    else if (size < THRESHOLD_2)
    {
        cout << "Selected method: "
             << "OpenMP QuickSort"
             << endl;

        cout << "OpenMP threads: "
             << OPENMP_THREADS
             << endl;

        cout << "Task cutoff: "
             << TASK_CUTOFF
             << endl;


        omp_set_num_threads(
            OPENMP_THREADS
        );

        auto start =
            high_resolution_clock::now();

        #pragma omp parallel
        {
            #pragma omp single
            {
                parallelQuickSort(
                    array,
                    0,
                    size - 1
                );
            }
        }

        auto stop =
            high_resolution_clock::now();

        auto duration =
            duration_cast<microseconds>(
                stop - start
            );

        cout << "Execution time: "
             << duration.count()
             << " microseconds"
             << endl;
    }

    // HYBRID
    else
    {
        cout << "Selected method: "
             << "Hybrid OpenCL + OpenMP"
             << endl;

        cout << "OpenMP threads: "
             << OPENMP_THREADS
             << endl;


        omp_set_num_threads(
            OPENMP_THREADS
        );


        // Setup OpenCL before timing.
        setup_openCL_device_context_queue_kernel(
            (char *)"./quicksort.cl",
            (char *)"quicksort_chunks"
        );

        printDeviceInfo();

        size_t globalWorkSize =
            GLOBAL_WORK_SIZE;

        size_t localWorkSize =
            LOCAL_WORK_SIZE;

        unsigned long chunkSize = (size + globalWorkSize - 1) / globalWorkSize;

        bufArray =
            clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                size * sizeof(int),
                NULL,
                &err
            );

        if (err < 0)
        {
            cout << "Couldn't create OpenCL buffer."
                 << endl;

            free(array);

            return 1;
        }

        cl_ulong deviceSize =
            (cl_ulong)size;

        cl_ulong deviceChunkSize =
            (cl_ulong)chunkSize;

        clSetKernelArg(
            kernel,
            0,
            sizeof(cl_mem),
            &bufArray
        );

        clSetKernelArg(
            kernel,
            1,
            sizeof(cl_ulong),
            &deviceSize
        );

        clSetKernelArg(
            kernel,
            2,
            sizeof(cl_ulong),
            &deviceChunkSize
        );

        cout << "Global work size: "
             << globalWorkSize
             << endl;

        cout << "Local work size: "
             << localWorkSize
             << endl;

        cout << "Chunk size: "
             << chunkSize
             << endl;

        // Start Hybrid timing.
        auto totalStart =
            high_resolution_clock::now();


        auto writeStart =
            high_resolution_clock::now();

        clEnqueueWriteBuffer(
            queue,
            bufArray,
            CL_TRUE,
            0,
            size * sizeof(int),
            array,
            0,
            NULL,
            NULL
        );

        auto writeStop =
            high_resolution_clock::now();

        // OpenCL GPU sorting
        auto kernelStart =
            high_resolution_clock::now();

        clEnqueueNDRangeKernel(
            queue,
            kernel,
            1,
            NULL,
            &globalWorkSize,
            &localWorkSize,
            0,
            NULL,
            NULL
        );

        clFinish(
            queue
        );

        auto kernelStop =
            high_resolution_clock::now();

        auto readStart =
            high_resolution_clock::now();

        clEnqueueReadBuffer(
            queue,
            bufArray,
            CL_TRUE,
            0,
            size * sizeof(int),
            array,
            0,
            NULL,
            NULL
        );

        auto readStop =
            high_resolution_clock::now();

        // OpenMP parallel merge
        auto mergeStart =
            high_resolution_clock::now();

        mergeSortedChunksOpenMP(
            array,
            size,
            chunkSize
        );

        auto mergeStop =
            high_resolution_clock::now();


        auto totalStop =
            high_resolution_clock::now();

        auto writeDuration =
            duration_cast<microseconds>(
                writeStop - writeStart
            );

        auto kernelDuration =
            duration_cast<microseconds>(
                kernelStop - kernelStart
            );

        auto readDuration =
            duration_cast<microseconds>(
                readStop - readStart
            );

        auto mergeDuration =
            duration_cast<microseconds>(
                mergeStop - mergeStart
            );

        auto totalDuration =
            duration_cast<microseconds>(
                totalStop - totalStart
            );

        cout << "Host to GPU time: "
             << writeDuration.count()
             << " microseconds"
             << endl;

        cout << "Kernel execution time: "
             << kernelDuration.count()
             << " microseconds"
             << endl;

        cout << "GPU to Host time: "
             << readDuration.count()
             << " microseconds"
             << endl;

        cout << "OpenMP merge time: "
             << mergeDuration.count()
             << " microseconds"
             << endl;

        cout << "Execution time: "
             << totalDuration.count()
             << " microseconds"
             << endl;

        free_opencl_memory();
    }

    // CORRECTNESS CHECK
    long long checksumAfter = calculateChecksum(array, size);

    if (
        checkSorted(
            array,
            size
        )
    )
    {
        cout << "Array sorted correctly."
             << endl;
    }
    else
    {
        cout << "Array sorting failed."
             << endl;
    }

    if (
        checksumBefore == checksumAfter
    )
    {
        cout << "Checksum matched."
             << endl;
    }
    else
    {
        cout << "Checksum failed."
             << endl;
    }

    free(
        array
    );

    return 0;
}

// RANDOM ARRAY
void randomArray(
    int array[],
    unsigned long size
)
{
    for (
        unsigned long i = 0;
        i < size;
        i++
    )
    {
        array[i] =
            rand();
    }
}

// PARTITION
int partition(
    int array[],
    int low,
    int high
)
{
    int pivot =
        array[high];


    int i =
        low - 1;


    for (
        int j = low;
        j < high;
        j++
    )
    {
        if (
            array[j] <=
            pivot
        )
        {
            i++;


            int temp =
                array[i];

            array[i] =
                array[j];

            array[j] =
                temp;
        }
    }


    int temp =
        array[i + 1];

    array[i + 1] =
        array[high];

    array[high] =
        temp;


    return i + 1;
}

// SEQUENTIAL QUICKSORT
void quickSort(
    int array[],
    int low,
    int high
)
{
    if (low < high)
    {
        int pivotIndex =
            partition(
                array,
                low,
                high
            );


        quickSort(
            array,
            low,
            pivotIndex - 1
        );


        quickSort(
            array,
            pivotIndex + 1,
            high
        );
    }
}

// OPENMP QUICKSORT
void parallelQuickSort(
    int array[],
    int low,
    int high
)
{
    if (low >= high)
    {
        return;
    }

    int size =
        high - low + 1;

    if (
        size <=
        TASK_CUTOFF
    )
    {
        quickSort(
            array,
            low,
            high
        );


        return;
    }

    int pivotIndex =
        partition(
            array,
            low,
            high
        );

    #pragma omp task shared(array)
    {
        parallelQuickSort(
            array,
            low,
            pivotIndex - 1
        );
    }

    #pragma omp task shared(array)
    {
        parallelQuickSort(
            array,
            pivotIndex + 1,
            high
        );
    }


    #pragma omp taskwait
}

// CHECK 
bool checkSorted(
    int array[],
    unsigned long size
)
{
    for (
        unsigned long i = 1;
        i < size;
        i++
    )
    {
        if (
            array[i - 1] >
            array[i]
        )
        {
            return false;
        }
    }


    return true;
}

// CHECKSUM
long long calculateChecksum(
    int array[],
    unsigned long size
)
{
    long long checksum =
        0;


    for (
        unsigned long i = 0;
        i < size;
        i++
    )
    {
        checksum +=
            array[i];
    }


    return checksum;
}

// OPENMP MERGE
void mergeSortedChunksOpenMP(
    int array[],
    unsigned long size,
    unsigned long chunkSize
)
{
    int *temp =
        (int *) malloc(
            size * sizeof(int)
        );

    unsigned long width =
        chunkSize;

    while (
        width <
        size
    )
    {
        #pragma omp parallel for schedule(static)
        for (
            long long left = 0;
            left < (long long)size;
            left +=
                (long long)(
                    2 * width
                )
        )
        {
            unsigned long middle =
                (unsigned long)left +
                width;

            unsigned long right =
                (unsigned long)left +
                2 * width;

            if (
                middle >
                size
            )
            {
                middle =
                    size;
            }

            if (
                right >
                size
            )
            {
                right =
                    size;
            }

            unsigned long i =
                (unsigned long)left;

            unsigned long j =
                middle;

            unsigned long k =
                (unsigned long)left;

            while (
                i < middle &&
                j < right
            )
            {
                if (
                    array[i] <=
                    array[j]
                )
                {
                    temp[k++] =
                        array[i++];
                }
                else
                {
                    temp[k++] =
                        array[j++];
                }
            }

            while (
                i < middle
            )
            {
                temp[k++] =
                    array[i++];
            }

            while (
                j < right
            )
            {
                temp[k++] =
                    array[j++];
            }

            for (
                unsigned long x =
                    (unsigned long)left;
                x < right;
                x++
            )
            {
                array[x] =
                    temp[x];
            }
        }

        width = width * 2;
    }

    free(
        temp
    );
}

// OPENCL SETUP
void setup_openCL_device_context_queue_kernel(
    char *filename,
    char *kernelname
)
{
    device_id =
        create_device();


    context =
        clCreateContext(
            NULL,
            1,
            &device_id,
            NULL,
            NULL,
            &err
        );


    if (err < 0)
    {
        cout << "Couldn't create OpenCL context."
             << endl;

        exit(1);
    }


    program =
        build_program(
            context,
            device_id,
            filename
        );


    queue =
        clCreateCommandQueueWithProperties(
            context,
            device_id,
            0,
            &err
        );


    if (err < 0)
    {
        cout << "Couldn't create command queue."
             << endl;

        exit(1);
    }


    kernel =
        clCreateKernel(
            program,
            kernelname,
            &err
        );


    if (err < 0)
    {
        cout << "Couldn't create OpenCL kernel."
             << endl;

        exit(1);
    }
}

// OPENCL BUILD PROGRAM
cl_program build_program(
    cl_context ctx,
    cl_device_id dev,
    const char *filename
)
{
    FILE *program_handle;

    char *program_buffer;
    char *program_log;

    size_t program_size;
    size_t log_size;

    program_handle =
        fopen(
            filename,
            "r"
        );

    if (program_handle == NULL)
    {
        perror("Couldn't find the kernel file");

        exit(1);
    }

    fseek(
        program_handle,
        0,
        SEEK_END
    );

    program_size =
        ftell(
            program_handle
        );

    rewind(
        program_handle
    );

    program_buffer =
        (char *) malloc(
            program_size + 1
        );

    program_buffer[program_size] =
        '\0';

    fread(
        program_buffer,
        sizeof(char),
        program_size,
        program_handle
    );

    fclose(
        program_handle
    );

    cl_program newProgram =
        clCreateProgramWithSource(
            ctx,
            1,
            (const char **)&program_buffer,
            &program_size,
            &err
        );

    free(
        program_buffer
    );

    err =
        clBuildProgram(
            newProgram,
            0,
            NULL,
            NULL,
            NULL,
            NULL
        );

    if (err < 0)
    {
        clGetProgramBuildInfo(
            newProgram,
            dev,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &log_size
        );

        program_log =
            (char *) malloc(
                log_size + 1
            );

        program_log[log_size] =
            '\0';

        clGetProgramBuildInfo(
            newProgram,
            dev,
            CL_PROGRAM_BUILD_LOG,
            log_size,
            program_log,
            NULL
        );

        cout << program_log
             << endl;

        free(
            program_log
        );

        exit(1);
    }

    return newProgram;
}

// OPENCL GPU DEVICE
cl_device_id create_device()
{
    cl_uint platformCount = 0;


    clGetPlatformIDs(
        0,
        NULL,
        &platformCount
    );


    cl_platform_id *platforms =
        (cl_platform_id *) malloc(
            platformCount *
            sizeof(cl_platform_id)
        );


    clGetPlatformIDs(
        platformCount,
        platforms,
        NULL
    );


    cl_device_id dev = NULL;


    for (cl_uint i = 0; i < platformCount; i++)
    {
        err =
            clGetDeviceIDs(
                platforms[i],
                CL_DEVICE_TYPE_GPU,
                1,
                &dev,
                NULL
            );


        if (
            err ==
            CL_SUCCESS
        )
        {
            platform =
                platforms[i];


            free(
                platforms
            );


            return dev;
        }
    }

    free(
        platforms
    );

    cout << "GPU OpenCL device not found."
         << endl;


    exit(1);
}

// OPENCL DEVICE INFORMATION
void printDeviceInfo()
{
    char deviceName[256];

    char deviceVendor[256];

    cl_device_type deviceType;

    clGetDeviceInfo(
        device_id,
        CL_DEVICE_NAME,
        sizeof(deviceName),
        deviceName,
        NULL
    );

    clGetDeviceInfo(
        device_id,
        CL_DEVICE_VENDOR,
        sizeof(deviceVendor),
        deviceVendor,
        NULL
    );

    clGetDeviceInfo(
        device_id,
        CL_DEVICE_TYPE,
        sizeof(deviceType),
        &deviceType,
        NULL
    );

    cout << "OpenCL Device: "
         << deviceName
         << endl;

    cout << "Device Type: "
         << (
                (
                    deviceType &
                    CL_DEVICE_TYPE_GPU
                )
                ? "GPU"
                : "CPU"
            )
         << endl;

    cout << "Device Vendor: "
         << deviceVendor
         << endl;
}

// RELEASE MEMORY
void free_opencl_memory()
{
    clReleaseMemObject(
        bufArray
    );


    clReleaseKernel(
        kernel
    );


    clReleaseCommandQueue(
        queue
    );


    clReleaseProgram(
        program
    );


    clReleaseContext(
        context
    );
}