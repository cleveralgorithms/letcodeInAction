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

## 171. Excel Sheet Column Number
- 2019-03-23

周六，写一个easy题，这题是让我们把Excel中的列序数如A、AB、ZY、ABC等转为数值；相当于从A映射到0，然后B、C这样累加上去，
分析发现就是把26进制转化为10进制；例如12相比于2是增加了一个分位，加了10^1。32比12增加了(3-2)\*10;302比2增加了10^(2);而AA比A增加了26;
再加上最近复习了一下Python内置的`ord()`函数，测试有ord('A')=65,于是从A映射到1就直接用`ord('A')-64`，于是有：
```python
class Solution:
    def titleToNumber(self, s: str) -> int:
        n=list(s)
        m=n[::-1] #翻转序列
        c=0
        for i in range(len(m)):
            if i==0:
                c=ord(m[i])-64 #A~Z 个位数的情况
            else:
                c+=(26**i)*(ord(m[i])-64) #其他位的情况
        return c
```
## 151. Reverse Words in a String
- 2019-03-24

还是写字符串类型的题目。151这题的标注是Medium难度，而且通过率不足20%，但看了描述之后觉得用Python实现不困难，代码如下，可以写得很简洁：
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        slst=s.split()
        res=slst[::-1] #翻转列表
        return ' '.join(res) #列表组合为字符串
```
## 620. Not Boring Movies
- 2019-03-25

**Tag**:SQL;

周一写个SQL题；选了一个easy题去写；注意not like就OK；

```mysql
select * from cinema where description not like "boring" and id % 2 != 0 order by rating desc;
```

## 215. Kth Largest Element in an Array
- 2019-03-26

求第k个最大的数，这题比较有趣的是可以有很多解法，可以暴力先排序，作为高频面试题，这种第K个的问题在数组很大的情况下用**堆**是标准答案，讨论区有个很好的解析文章，好好去理解。
```python
import heapq
class Solution(object):
    def findKthLargest(self, nums, k):
        nums = [-num for num in nums]
        heapq.heapify(nums)
        res = float('inf')
        for _ in range(k):
            res = heapq.heappop(nums)
        return -res
```
-[ ] 去消化[ Python | 给你把这道题讲透](https://leetcode.com/problems/kth-largest-element-in-an-array/discuss/167837/Python-or-tm)

## 160. Intersection of Two Linked Lists
- 2019-03-27

找到链表的第一个交点，刚开始确实不会优秀的解法，于是复习了一下链表的高频考题：链表是否有环、链表是否相交、求第一个交点等；这题的一种解法是：

> 采用对齐的思想。计算两个链表的长度 L1 , L2，分别用两个指针 p1 , p2 指向两个链表的头，然后将较长链表的 p1（假设为 p1）向后移动L2 - L1个节点，然后再同时向后移动p1 , p2，直到 p1 = p2。相遇的点就是相交的第一个节点。

我看讨论区还有用栈的，全部入栈然后pop出来，也是一种解法。
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """

        l1,l2=0,0
        h1=headA
        h2=headB
        while h1:
            l1+=1
            h1=h1.next
        while h2:
            l2+=1
            h2=h2.next
        p1=headA
        p2=headB
        if l1<l2:
            for i in range(l2-l1):
                p2=p2.next
        else:
            for i in range(l1-l2):
                p1=p1.next
        
        while p1:
            if p1==p2:
                return p1
            else:
                p1=p1.next
                p2=p2.next
        return None

```

## 21. Merge Two Sorted Lists
- 2019-03-28

合并两个链表，用常规思路就是用好if判断，看到一个不错的答案用到了递归，值得学习
```python
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1 or not l2:
            return l1 or l2
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2 
```
## 169. Majority Element
- 2019-03-29

找出数组中出现次数大于一半的元素，这个我觉得那个O(n)的解法很有价值，因此按照这个去学习，讨论区Python最高赞是用sorted排序，很优雅，但从算法来说，Moore's voting algorithm更值得消化。`return sorted(num)[len(num)/2]`。

```python
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # implement the Moore's voting algorithm: find a pair different element and delete it
        count = 0
        for i in range(0, len(nums)):
            if count == 0:
                key = nums[i]
                count = 1
            else:
                if key == nums[i]:
                    count += 1
                else:
                    count -= 1
        return key
```
## 3. Longest Substring Without Repeating Characters
- 2019-03-30

最长不重复子串。参考了Solution里面的思想。

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start = maxLength = 0 #初始值
        usedChar = {}
        
        for i in range(len(s)):
            if s[i] in usedChar and start <= usedChar[s[i]]:
                start = usedChar[s[i]] + 1
            else:
                if maxLength <1+i-start:
                    maxLength = 1+i-start

            usedChar[s[i]] = i

        return maxLength
```












