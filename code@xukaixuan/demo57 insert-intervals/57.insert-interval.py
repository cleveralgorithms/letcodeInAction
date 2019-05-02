#
# @lc app=leetcode id=57 lang=python3
#
# [57] Insert Interval
#
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        s,e = newInterval[0],newInterval[-1]
        left = [i for i in intervals if i[-1]<s]
        right = [i for i in intervals if i[0]>e]
        if left + right != intervals:
            s = min(s,intervals[len(left)][0])
            e = max(e,intervals[~len(right)][-1])
        return left + [Interval(s,e)] +right
        

