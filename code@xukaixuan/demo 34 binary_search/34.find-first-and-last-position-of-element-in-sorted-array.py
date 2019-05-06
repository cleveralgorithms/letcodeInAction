#
# @lc app=leetcode id=34 lang=python
#
# [34] Find First and Last Position of Element in Sorted Array
#
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        lo,hi = 0,len(nums)-1
        first,last = -1,-1
        while(lo <=hi):
            mid = lo +( hi -lo)//2
            if target< nums[mid]:
                hi = mid -1
            elif target >  nums[mid]:
                lo = mid + 1
            else :
                if (mid == 0 or nums[mid-1] < target):
                    first = mid
                hi = mid-1
        lo,hi = 0,len(nums)-1
        while(lo <= hi):
            mid = lo +( hi -lo)//2
            if target< nums[mid]:
                hi = mid -1
            elif target >  nums[mid]:
                lo = mid + 1
            else :
                if (mid == len(nums)-1 or nums[mid+1] > target):
                    last = mid
                lo = mid+1
        return [first,last]


## most voted 

def searchRange(self, nums, target):
    def binarySearchLeft(A, x):
        left, right = 0, len(A) - 1
        while left <= right:
            mid = (left + right) / 2
            if x > A[mid]: left = mid + 1
            else: right = mid - 1
        return left

    def binarySearchRight(A, x):
        left, right = 0, len(A) - 1
        while left <= right:
            mid = (left + right) / 2
            if x >= A[mid]: left = mid + 1
            else: right = mid - 1
        return right
        
    left, right = binarySearchLeft(nums, target), binarySearchRight(nums, target)
    return (left, right) if left <= right else [-1, -1]

        
          

        

