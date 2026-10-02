long partition_array(
    __global int *array,
    long low,
    long high
)
{
    int pivot = array[high];

    long i = low - 1;


    for (long j = low; j < high; j++)
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

__kernel void quicksort_chunks(
    __global int *array,
    ulong size,
    ulong chunkSize
)
{
    ulong id = get_global_id(0);

    long low = (long)(id * chunkSize);

    long high = (long)(low + chunkSize - 1);

    if (low >= size)
        return;

    if (high >= size)
        high = size - 1;

    long lowStack[128];
    long highStack[128];

    int top = -1;

    while (1)
    {
        while (low < high)
        {
            long pivot = partition_array(array, low, high);

            long leftSize = pivot - low;

            long rightSize = high - pivot;


            if (leftSize < rightSize)
            {
                if (pivot + 1 < high)
                {
                    top++;
                    lowStack[top] = pivot + 1;

                    highStack[top] = high;
                }
                high = pivot - 1;
            }
            else
            {
                if (low < pivot - 1)
                {
                    top++;
                    lowStack[top] = low;
                    highStack[top] = pivot - 1;
                }
                low = pivot + 1;
            }
        }


        if (top < 0)
            break;

        low = lowStack[top];

        high = highStack[top];
        top--;
    }
}
