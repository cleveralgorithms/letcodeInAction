"""

"""
def merge_sort(arr):
    if len(arr)<= 1:
        return arr
    mid  = len(arr)//2
    left,right = merge_sort(arr[:mid]),merge_sort(arr[mid:])
    return merge(left,right,arr.copy())

def merge(left,right,merged):
    l_cur,r_cur = 0,0
    while l_cur<len(left) and r_cur <len(right):
        if left[l_cur]<= right[r_cur]:
            merged[l_cur+r_cur] = left[l_cur]
            l_cur+=1
        else:
            merged[l_cur+r_cur] = right[r_cur]
            r_cur+=1
        for l_cur in range(l_cur,len(left)):
            merged[l_cur+r_cur]=left[l_cur]
        for r_cur in range(r_cur,len(right)):
            merged[l_cur+r_cur]=right[r_cur]
    return merged
