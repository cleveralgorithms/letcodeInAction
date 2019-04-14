def bubble_sort(arr):
    swapped = True
    x = -1
    while swapped:
        swap = False
        x += 1
        for i in range(1, n - x):
            if arr[i - 1] > arr[i]:
                arr[i - 1], arr[i] = arr[i], arr[i - 1]
                swapped = True

    return arr


def test():
    print(bubble_sort([1, 3, 4, 6, 0, 1, 4, 2]))


test()
