#
# @lc app=leetcode id=56 lang=python3
#
# [56] Merge Intervals
#
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        ans = []
        for invt in sorted(intervals,key = lambda x :x[0]):
            if (ans and ans[-1][-1] >= invt[0]):
                ans[-1][-1] = max(invt[-1],ans[-1][-1])
            else:
                ans.append(invt)
        return ans


        

