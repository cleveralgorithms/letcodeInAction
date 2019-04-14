def select_sort(arr):
    for i in range(len(arr)):
       minum = i
       for j in range(i+1,len(arr)):
           if arr[j] < arr(minum):
               minum = j
        arr[minum] ,arr[i] = arr[i],arr[minum]
     return arr

print(select_sort([0,1,8,3,4,6,3,5]))
