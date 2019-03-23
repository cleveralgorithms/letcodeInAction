# 方法一：
# 两层循环暴力解决，符合直觉，缺点是空间复杂度O(n2)

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        count = len(nums)
        for i in range(count-1):  # 第一层循环不包括nums最后一个元素
            for j in range(i+1, count):
                if (nums[i] + nums[j]) == target:
                    return [i,j]

# 方法二：
# 抄答案，建立哈希表时间复杂度O(n),查询时间复杂度O(1)
# mapping={} + 内建序列函数 enumerate 实现简洁追踪元素索引，
# 

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}
        for index, num in enumerate(nums):
            another_num = target - num
            if another_num in hashmap:
                return [hashmap[another_num], index] # 注意这里的顺序
            hashmap[num] = index
        return None