lcfleetcode

## 前言
随着leetcode进入千题时代，刷题的姿势对效果的作用也是很大的；打算按着知识点类型进行划分，不按照题目序号刷

## 1. two-sum 
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
## 7. Reverse Integer
- 2019-03-31

直接利用列表的参数去逆转：
```python
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        res=0
        if x<0:
            res=-int(str(x)[::-1][:-1])
        else:
            res=int(str(x)[::-1])
        #res=-int(str(x)[::-1][:-1]) if x<0 else int(str(x)[::-1])
        if res<-2**31 or res+1>2**31:
            return 0
        return res
```
还可以用栈去逆置，需要O(n)的额外空间，并且需要关注是否会栈溢出。。O(log(n))的做法。。

## 5. Longest Palindromic Substring
- 2019-03-31 还是31号。3月刷指标(:滑稽)

最长回文子串。暴力法很容易想，O(n^3)。判断回文需要n（可以到log(n)吧?）。可以用动规，需要O(n^2),同时需要O(n^2)的空间。

```python
class Solution(object):
    def longestPalindrome(self,s):
        res = ""
        for i in range(len(s)):        
            odd  = self.palindromeAt(s, i, i)
            even = self.palindromeAt(s, i, i+1)
        
            res = max(res, odd, even, key=len)
        return res
 
    # starting at l,r expand outwards to find the biggest palindrome
    def palindromeAt(self, s, l, r):    
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l+1:r]
```

## 4. Median of Two Sorted Arrays
- 2019-04-01

这是个hard题，因为刷一个每日签到到了这题。暴力法easy，要O(min(n,m))不容易。

参考：[Share-my-O(log(min(mn))-solution-with-explanation](https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2481/Share-my-O(log(min(mn))-solution-with-explanation)
```python
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        curr = prev = 0
        total = len(nums1) + len(nums2)
        nums = nums1 + list(reversed(nums2))
        while total - len(nums) < (1 + int(total / 2)):
            index = -1 if nums[0] > nums[-1] else 0
            prev, curr = curr, nums.pop(index)
        return curr if total % 2 else (prev + curr) / 2.0
```

## 46. Permutations
- 2019-04-02

输出数值型列表的全排列,递归的核心思路是**将每个元素放到余下n-1个元素组成的队列最前方，对剩余元素进行递归全排列**。今天整理了一篇笔记文章：[Ann全排列的枚举_递归实现(基于Python)@jianshu](https://www.jianshu.com/p/a5aed1bf5c80)

```python
class Solution(object):
    def permute(self, nums):
        """
        :type lst: List[int]
        :rtype: List[List[int]]
        """
        n=len(nums)
        if n<=1:
            return [nums]
        elif n==2:
            return [[nums[0],nums[1]],[nums[1],nums[0]]]
        kk=[]
        for i in range(n):
            nlst=nums[0:i]+nums[i+1:] 
            c=self.permute(nlst)
            ss=[]
            for j in c:
                w=[nums[i]]
                w.extend(j)
                ss.append(w)
            kk.extend(ss)
        return kk
```

## 2. Add Two Numbers

- 2019-04-02

正常解法：循环节点相加放到列表里再合并为一个新链表，O(max(m,n))，现在用Python解链表题还不够熟练,后续专题写一下链表题。在讨论区发现了一种“流氓”解法，写一个toint和tolist函数变成整数计算再解包为链表。

```python
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        addends = l1, l2
        dummy = end = ListNode(0)
        carry = 0
        while addends or carry:
            carry += sum(a.val for a in addends)
            addends = [a.next for a in addends if a.next]
            end.next = end = ListNode(carry % 10)
            carry /= 10
        return dummy.next

```

## 11. Container With Most Water
- 2019-04-02

容器最多能装多少水。

看讨论区大家的高赞解法是O(n)

```python

```
## 15. 3Sum
- 2019-04-03

从列表中选出三个数a，b，c，满足sum(a,b,c)==0。暴力法，三个循环，O(n^3)。用字典可以降低一层复杂度。讨论区大家普遍先排序。
```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        N=len(nums)
        res=[]
        for i in range(N):
            if i > 0 and nums[i] == nums[i-1]: #两个元素相同时，i再走一步
                continue
            target =-nums[i]
            s,e = i+1, N-1
            while s<e: #下面就是循环试 s+e=target  化归到2sum
                if nums[s]+nums[e] == target:
                    res.append([nums[i], nums[s], nums[e]])
                    s = s+1
                    while s<e and nums[s] == nums[s-1]:
                        s = s+1
                elif nums[s] + nums[e] < target:
                    s = s+1
                else:
                    e = e-1
        return res
```
## 20. Valid Parentheses
- 2019-04-04

合法的括号，很经典的用**栈**这种先进后出(FILO)数据结构的题目，思路就是对字符串`s`遍历，如果是左括号，入栈，遇到右括号，看栈顶(就是最新加入栈的元素)是否是对应的左括号，如果不是，直接是false，如果是，这个左括号出栈。当遍历完了栈正好空则true。同时考虑只有左或右括号的情况，考虑判断时栈为空的情况。
```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack=[] #用栈的做法
        left=['(','[','{']
        right=[')',']','}']
        for i in s:
            if i in left:
                stack.append(i)
            else:
                for j,v in enumerate(right):
                    if i==v:
                        if len(stack)==0:
                            return False
                        elif stack[-1]==left[j]: #peek
                            stack.pop()
                            break #break for j,v
                        else:
                            return False
        if len(stack)>0:
            return False
        return True #len==0 才是true
```
## 17. Letter Combinations of a Phone Number
- 2019-04-05 清明

也算是一个排列组合题，可以用回溯法解。

```python
class Solution(object): #暴力法
    def letterCombinations(self, digits):#digits: str
        phone = {'2': ['a', 'b', 'c'],'3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],'5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],'7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],'9': ['w', 'x', 'y', 'z']} #忽略1
        if not digits:
            return []
        res=phone[digits[0]]
        for d in digits[1:]:
            ks=phone[d]
            nres=[]
            for i in ks:
                nres.extend([r+i for r in res]) #感觉比下面更高效些
            #for r in res:nres.extend([r+i for i in ks])
            res=nres
        return res
```
上面的解法：
Runtime: 36 ms, faster than 8.86%；Memory Usage: 12.2 MB, less than 5.15%；
用官方的回溯法的话:Runtime: 24 ms, faster than 26.82%； Memory Usage: 12.2 MB, less than 5.15%； 内存用量相同，速度更快些。回溯法是O((3^Nx4^M)的时间复杂度。
```python
class Solution(object):
    def letterCombinations(self, digits):
        phone = {'2': ['a', 'b', 'c'],'3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],'5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],'7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],'9': ['w', 'x', 'y', 'z']} #忽略1
        def backtrack(combination, next_digits):
            # if there is no more digits to check
            if len(next_digits) == 0:
                # the combination is done
                output.append(combination)
            # if there are still digits to check
            else:
                # iterate over all letters which map 
                # the next available digit
                for letter in phone[next_digits[0]]:
                    # append the current letter to the combination
                    # and proceed to the next digits
                    backtrack(combination + letter, next_digits[1:])
                    
        output = []
        if digits:
            backtrack("", digits)
        return output
```
## 29. Divide Two Integers
- 2019-04-06

不用乘法、除法符号求两个数相除的商，能用的就是加减和位运算了。（Python的）位运算需要去复习。
```python
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        positive = (dividend < 0) is (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        while dividend >= divisor:
            temp, i = divisor, 1
            while dividend >= temp:
                dividend -= temp
                res += i
                i <<= 1
                temp <<= 1
        if not positive:
            res = -res
        return min(max(-2147483648, res), 2147483647)
```
## 26. Remove Duplicates from Sorted Array
- 2019-04-07

这题要求比较多，在O(1)的额外空间使用下，修改数组的前面变成没有重复值的数组，同时不管后面的值了，返回是没有重复值的数组的长度。用set(nums)居然无效。
不过双指针法也不难理解。
```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        newTail = 0
        #用两个指针，一个从0慢慢走，修改走到的位置，一个往前走
        for i in range(1, len(nums)):
            if nums[i] != nums[newTail]:
                newTail += 1
                nums[newTail] = nums[i]
        return newTail + 1
```
## 33. Search in Rotated Sorted Array
- 2019-04-08

在一个经过了旋转的数组中搜索目标元素，要求在O(log(n))实现，显然需要二分搜索
```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums:
            return -1

        low, high = 0, len(nums) - 1

        while low <= high:
            mid = (low + high) / 2
            if target == nums[mid]:
                return mid

            if nums[low] <= nums[mid]:
                if nums[low] <= target <= nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] <= target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1

        return -1
```



## 42. Trapping Rain Water

这题2018年在面试中遇到过，当时想了好久才推出递归写法，早就后悔没多刷题了，看到这题更后悔没早点多刷题了。
```python
# 等我找到我以前的解法

```



逆序的好处：方便add；
顺序的好处：翻下来的时候可以顺路看之前的笔记，顺序符合心理认知。







