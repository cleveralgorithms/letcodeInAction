lcfleetcode

## 前言
随着leetcode进入千题时代，刷题的姿势对效果的作用也是很大的；打算按着知识点类型进行划分，不按照题目序号刷

## two-sum 
- 2019-03-20
tag:**math**
求列表中能组合出目标值的两个数；作为经典题，暴力法只需要能枚举列表中的两两组合就好；

```python
#Brute Force
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nc=len(nums)
        for i in range(0,nc-1):
            for j in range(i+1,nc):
                if nums[i]+nums[j]==target:
                    return [i,j]

```
用一个哈希表，用到Python强大的in
```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # a dictionary keeping the already visited values
        numd = {}
        for i in range(len(nums)):
            value = nums[i]
            # complementary value which we are searching for
            comp = target - value
            if (comp in numd):
                # we found the complement: return the answer
                return [numd[comp], i]
            else:
                # we did not find any complements, let's keep this item for later
                numd[value] = i

        return [-1,-1] 
```
## 58. Length of Last Word
- 2019-03-21
tag: **string**
题目要求返回一个字符串最后一个单词的长度 正常的字符串如"a boy"等都很容易，题目说如果最后一个词不存在则返回0；我测试了两次了解到这些对应的测试用例类似于： 'a '、' '、'   '；在Python中处理字符串很容易，字符串切分就用`split(' ')`，我的思路是清除最后的空格，如果切分后还是'',说明应该返回0；否则就是正常的len；代码如下：
```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        s=s.strip()
        if s=="":
            return 0
        w=len(s.split(' ')[-1])
        return w
```

## 197. Rising Temperature
- 2019-03-22
tag：**SQL**
周五了写一个SQL题，自己的SQL需要强化呀。题目要求找出当天温度比前一天高的日期id；有很多实现方式，
不建新列的话，思路基本都是把Weather表当两个表用，可以用join，或者直接当两个表比较，下面是一种实现：
```mysql
select a.Id as 'Id' from weather as a,weather as b
where a.Temperature>b.Temperature and datediff(a.RecordDate,b.RecordDate) =1;
```
*datediff*等处理日期的函数该深入学一下；

















