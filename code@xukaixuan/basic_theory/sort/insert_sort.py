def insert_sort(arr):
    """
    每次循环迭代中，插入排序从数组中删除一个元素。
    然后再一个已经有序的数组中找到该元素的位置，并将其插入其中。
    """
    for i in range(len(arr)):
        cur = arr[i]
        pos = i

        while pos >0 and arr[pos-1] > cur:
            arr[pos] = arr[pos-1]
            pos = pos -1
        arr[pos] = cur
    return arr
