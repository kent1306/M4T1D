#include <CL/cl.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <stdio.h>

using namespace std::chrono;
using namespace std;


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
void randomArray(int array[], unsigned long size);

bool checkSorted(
    int array[],
    unsigned long size
);

long long calculateChecksum(
    int array[],
    unsigned long size
);

void mergeSortedChunks(
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

void free_memory();

int main()
{
    unsigned long size;

    cout << "Enter array size: ";
    cin >> size;

    srand(42);

    int *array =
        (int *) malloc(size * sizeof(int));

    randomArray(array, size);

    long long checksumBefore =
        calculateChecksum(array, size);

    // Start OpenCL
    setup_openCL_device_context_queue_kernel(
        (char *)"./quicksort.cl",
        (char *)"quicksort_chunks"
    );

    size_t globalWorkSize = 256;
    size_t localWorkSize = 64;

    unsigned long chunkSize =
        (size + globalWorkSize - 1)
        / globalWorkSize;

    bufArray =
        clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            size * sizeof(int),
            NULL,
            &err
        );

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

    printDeviceInfo();

    auto start =
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


    // Run GPU kernel
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

    clFinish(queue);

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

    mergeSortedChunks(
        array,
        size,
        chunkSize
    );


    auto stop =
        high_resolution_clock::now();


    auto duration =
        duration_cast<microseconds>(
            stop - start
        );


    long long checksumAfter =
        calculateChecksum(array, size);


    cout << endl;

    cout << "Array size: "
         << size << endl;

    cout << "Global work size: "
         << globalWorkSize << endl;

    cout << "Local work size: "
         << localWorkSize << endl;

    cout << "Chunk size: "
         << chunkSize << endl;

    cout << "Execution time: "
         << duration.count()
         << " microseconds"
         << endl;


    if (checkSorted(array, size))
    {
        cout << "Array sorted correctly."
             << endl;
    }
    else
    {
        cout << "Array sorting failed."
             << endl;
    }


    if (checksumBefore == checksumAfter)
    {
        cout << "Checksum matched."
             << endl;
    }
    else
    {
        cout << "Checksum failed."
             << endl;
    }


    free(array);

    free_memory();

    return 0;
}


void randomArray(
    int array[],
    unsigned long size
)
{
    for (unsigned long i = 0; i < size; i++)
    {
        array[i] = rand();
    }
}


bool checkSorted(
    int array[],
    unsigned long size
)
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


long long calculateChecksum(
    int array[],
    unsigned long size
)
{
    long long checksum = 0;

    for (unsigned long i = 0; i < size; i++)
    {
        checksum += array[i];
    }

    return checksum;
}


void mergeSortedChunks(
    int array[],
    unsigned long size,
    unsigned long chunkSize
)
{
    int *temp =
        (int *) malloc(size * sizeof(int));

    unsigned long width =
        chunkSize;


    while (width < size)
    {
        for (unsigned long left = 0; left < size; left += 2 * width)
        {
            unsigned long middle = left + width;

            unsigned long right = left + 2 * width;


            if (middle > size)
                middle = size;

            if (right > size)
                right = size;


            unsigned long i = left;
            unsigned long j = middle;
            unsigned long k = left;


            while ( i < middle && j < right)
            {
                if (array[i] <= array[j])
                    temp[k++] = array[i++];
                else
                    temp[k++] = array[j++];
            }


            while (i < middle)
                temp[k++] = array[i++];


            while (j < right)
                temp[k++] = array[j++];


            for (
                unsigned long x = left;
                x < right;
                x++
            )
            {
                array[x] = temp[x];
            }
        }


        width = width * 2;
    }


    free(temp);
}


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

    kernel =
        clCreateKernel(
            program,
            kernelname,
            &err
        );
}

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
        fopen(filename, "r");

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

    program_size = ftell(program_handle);

    rewind(program_handle);

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

    fclose(program_handle);

    cl_program newProgram =
        clCreateProgramWithSource(
            ctx,
            1,
            (const char **)&program_buffer,
            &program_size,
            &err
        );

    free(program_buffer);

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

        cout << program_log << endl;

        free(program_log);

        exit(1);
    }

    return newProgram;
}


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

    // Search all platforms for a GPU.
    for (
        cl_uint i = 0;
        i < platformCount;
        i++
    )
    {
        err =
            clGetDeviceIDs(
                platforms[i],
                CL_DEVICE_TYPE_GPU,
                1,
                &dev,
                NULL
            );

        if (err == CL_SUCCESS)
        {
            platform = platforms[i];

            free(platforms);

            return dev;
        }
    }

    free(platforms);

    cout << "GPU OpenCL device not found."
         << endl;

    exit(1);
}

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
         << ((deviceType &
              CL_DEVICE_TYPE_GPU)
             ? "GPU"
             : "CPU")
         << endl;

    cout << "Device Vendor: "
         << deviceVendor
         << endl;
}

void free_memory()
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