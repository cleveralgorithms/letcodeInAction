def partition(array, begin, end):
    # 选取数组最后一位元素作为中转元素
    pivot = array[end]
    i = begin
    for j in range(begin, end):
        if array[j] < pivot:
            array[i], array[j] = array[j], array[i]
            i += 1
    array[i], array[end] = array[end], array[i]
    return i


def quick_sort_recursion(array, begin, end):
    if begin >= end:
        return
    pivot_idx = partition(array, begin, end)
    quick_sort_recursion(array,begin,pivot_idx-1)
    quick_sort_recursion(array,pivot_idx+1,end)


def quick_sort(arrar):
    return quick_sort_recursion(arrar,0,len(arrar)-1)
