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
## 34. Find First and Last Position of Element in Sorted Array
- 2019-04-09

排序了的数组，不用说太多，二分搜索，当然这题也有些小心机，处理好大于等于小于的情况，下面是官方代码：

```python
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        left_idx = self.extreme_insertion_index(nums, target, True)

        # assert that `left_idx` is within the array bounds and that `target`
        # is actually in `nums`.
        if left_idx == len(nums) or nums[left_idx] != target:
            return [-1, -1]
        return [left_idx, self.extreme_insertion_index(nums, target, False)-1]
    def extreme_insertion_index(self, nums, target, left):
        lo = 0
        hi = len(nums)

        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] > target or (left and target == nums[mid]):
                hi = mid
            else:
                lo = mid+1

        return lo
```
## 27. Remove Element
- 2019-04-10

把等于目标值的数从数组中原地删除，不能用超过O(n)的额外空间，这个的实现还是很值得思考的。当然实际中多用空间也不失为太差的解法，巨量数据时反复移动不一定比加到新数组优秀。
```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        nextPos = 0
        for num in nums:
            if num != val:
                nums[nextPos] = num
                nextPos += 1
        return nextPos
```
在评论区看到一种暴力方法，当然这种方法值得质疑和思考
```python
def removeElement(self, nums, val):
    try:
        while True:
            nums.remove(val)
    except:
        return len(nums)
```
> Since we don't know the implementation of the remove func, so you cannot make sure it fits the limitation of O(1) memory 
(不知道remove的机制，不能保证满足O(1)的空间限制)
> 列表每次Delete Item是O(n)，所以加上while会是O(n^2)，确实暴力。
## 39. Combination Sum
- 2019-04-11

一个需要遍历各种情况的题，比较好的解法是动规或深度优先搜索。
```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort() #动规
        dp = [[[]]] + [[] for i in xrange(target)]
        for i in xrange(1, target + 1):
            for number in candidates:
                if number > i: break
                for L in dp[i - number]:
                    if not L or number >= L[-1]: dp[i] += L + [number],
        return dp[target]
```
dfs的解法：
```python
def combinationSum(self, candidates, target):
    res = []
    candidates.sort()
    self.dfs(candidates, target, 0, [], res)
    return res
    
def dfs(self, nums, target, index, path, res):
    if target < 0:
        return  # backtracking
    if target == 0:
        res.append(path)
        return 
    for i in xrange(index, len(nums)):
        self.dfs(nums, target-nums[i], i, path+[nums[i]], res)
```
## 48. Rotate Image
- 2019-04-12

 in-place是值得思考的，暴力法是用O(N^2)的额外空间，我试了一下，是5.30%的空间优先率，很低了。
 ```python
 class Solution(object):
    def rotate(self, matrix):#暴力法
        nm=[]
        n=len(matrix) #i==j
        i,j=0,0
        for i in range(n):
            m=[matrix[j][i] for j in range(n-1,-1,-1)]
            nm.append(m)
        for i in range(n):
            for j in range(n):
                matrix[i][j]=nm[i][j]
    def rotate2n2(self,matrix):
        matrix[::] = zip(*matrix[::-1]) # matrix[::-1]->O(n^2), zip()-> O(n^2), 
        #although matrix[::] ask for an in-place replacement, take extra O(2n^2) extra space.
    def rotate2(self, matrix):
        n = len(matrix)#这种方法没去验证
        for l in xrange(n / 2):
            r = n - 1 - l
            for p in xrange(l, r):
                q = n - 1 - p #q=~p  
                cache = matrix[l][p]
                matrix[l][p] = matrix[q][l]
                matrix[q][l] = matrix[r][q]
                matrix[r][q] = matrix[p][r]
                matrix[p][r] = cache
 ```
## 37. Sudoku Solver
- 2019-04-13

必须独立解出来的题；
```python
class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        self.board = board
        self.solve()
    
    def findUnassigned(self):
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == ".":
                    return row, col
        return -1, -1
    
    def solve(self):
        row, col = self.findUnassigned()
        #no unassigned position is found, puzzle solved
        if row == -1 and col == -1:
            return True
        for num in ["1","2","3","4","5","6","7","8","9"]:
            if self.isSafe(row, col, num):
                self.board[row][col] = num
                if self.solve():
                    return True
                self.board[row][col] = "."
        return False
            
    def isSafe(self, row, col, ch):
        boxrow = row - row%3
        boxcol = col - col%3
        if self.checkrow(row,ch) and self.checkcol(col,ch) and self.checksquare(boxrow, boxcol, ch):
            return True
        return False
    
    def checkrow(self, row, ch):
        for col in range(9):
            if self.board[row][col] == ch:
                return False
        return True
    
    def checkcol(self, col, ch):
        for row in range(9):
            if self.board[row][col] == ch:
                return False
        return True
       
    def checksquare(self, row, col, ch):
        for r in range(row, row+3):
            for c in range(col, col+3):
                if self.board[r][c] == ch:
                    return False
        return True
```
## 47. Permutations II
- 2019-04-14

和46题的区别是这次输入可以包含重复的数字，例如可以输入`[1,1,2]`，有更巧妙的解法，但暴力法是用46题的递归然后对生成的结果去重（最后去重或每次递归都去重）

```python
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n=len(nums)
        if n<=1:
            return [nums]
        elif n==2: #注意这部分输入的坑
            if nums[0]==nums[1]:
                return [nums]
            return [[nums[0],nums[1]],[nums[1],nums[0]]]
        kk=[]
        for i in range(n):
            nlst=nums[0:i]+nums[i+1:] 
            c=self.permuteUnique(nlst)
            ss=[]
            for j in c:
                w=[nums[i]]
                w.extend(j)
                ss.append(w)
            kk.extend(ss)
            ks=[]
            for k in kk:
                if k not in ks:
                    ks.append(k)
            kk=ks
        return kk
```
## 50. Pow(x, n)
- 2019-04-15

自己去实现乘方。因为输入的n是包含负数且到2^31-1，还是需要考虑边界和超限的，在Python里这些问题比较容易解决。说回实现上，暴力法是直接循环n次算乘法，我知道还有递推公式，只记得了偶数`x^n=x^{n/2}*x^{n/2}`；奇数`x^n=x^{n/2}*x^{n/2}*x`；在《编程之美》里还有迭代法，之后再研究。
```python
class Solution(object):
    def myPow(self, x, n):
        if n==0:
            return 1
        elif n==1:
            return x
        npow=True if n<0 else False
        if npow: #负数的情况，只需要在递归的最后一次变成倒数就好，递归里面用正数算
            n=-n
            kx=self.myPow(x,n//2)
            if n%2==0:#偶数
                return 1/(kx*kx)
            else:
                return 1/(kx*kx*x)
        else:
            kx=self.myPow(x,n//2)
            if n%2==0:#偶数
                return kx*kx
            else:
                return kx*kx*x
```
## 35. Search Insert Position
- 2019-04-16

排序数组，显然最优是用二分；

```python
class Solution(object):
    def searchInsert(self, nums, target):
        if target > nums[len(nums) - 1]:
            return len(nums)

        if target < nums[0]:
            return 0

        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r)/2
            if nums[m] > target:
                r = m - 1
                if r >= 0:
                    if nums[r] < target:
                        return r + 1
                else:
                    return 0

            elif nums[m] < target:
                l = m + 1
                if l < len(nums):
                    if nums[l] > target:
                        return l
                else:
                    return len(nums)
            else:
                return m
```
在评论区发现一些很神奇的解法：一行：`return len([x for x in nums if x<target])`  O(n);
```python
try:
    return nums.index(target)
except:
    nums.append(target)
    nums.sort() #O(nlog(n))
    return nums.index(target)
```
## 19. Remove Nth Node From End of List
- 2019-04-17

链表题，复杂度不高，但要实现很高效也比较有挑战

```python
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        fast = slow = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head
```
## 56. Merge Intervals
- 2019-04-18

合并区间，不排序的话就需要每次比较时往前看（一个while或for循环在一个for循环里），需要O(n^2)，因此还是排序吧，根据首个元素排序后的判断就容易了
```python
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if not intervals:
            return []
        intervals=sorted(intervals,key=lambda j:j[0])
        res=[]
        cur=intervals[0]
        res.append(cur)
        for i in intervals:
            if res[-1][1]>=i[0] and res[-1][1]<i[1]:
                res[-1][1]=i[1]
            elif res[-1][1]<i[0]:
                res.append(i)
            else:
                pass
        return res
```
官方解法，list.start这种写法在自己的编译器上运行不了，查2.7.1和3.7的官方文档都没有。`AttributeError: 'list' object has no attribute 'start'`

```python
class Solution:#官方解法
    def merge(self, intervals):
        intervals.sort(key=lambda x: x.start)
        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1].end < interval.start:
                merged.append(interval)
            else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
                merged[-1].end = max(merged[-1].end, interval.end)
        return merged
```
## 59. Spiral Matrix II
- 2019-04-18

这次不是展开螺旋矩阵了，是生成；
```python
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        A = [[0] * n for _ in range(n)]
        i, j, di, dj = 0, 0, 0, 1
        for k in range(n*n):
            A[i][j] = k + 1
            if A[(i+di)%n][(j+dj)%n]:
                di, dj = dj, -di
            i += di
            j += dj
        return A
```
## 38. Count and Say
- 2019-04-19

这是个easy题，还是很巧妙的，计算n时n-1情况下值的计数形成一个新数（返回值为str类型），n取决于n-1，所以有递推，推导出递推式子后写递归。关键是统计出连续的值有多少个，在连续时，count，不连续时换新的count。
```python
class Solution(object):
    def countAndSay(self, n):
        if n==1: #终止条件
            return '1'
        res=''
        nm=self.countAndSay(n-1)
        k=len(nm)
        cv=[nm[0],0]  #当前计数的值 可以用dict {nm[0]:0}
        for i in nm:
            if i==cv[0]:
                cv[1]=cv[1]+1
            else: #不连续时，记录之前的状态+重置cv
                res='{0}{1}{2}'.format(res,cv[1],cv[0])
                cv=[i,1]
        res='{0}{1}{2}'.format(res,cv[1],cv[0])
        return res

```
## 62. Unique Paths
- 2019-04-20

这是很有趣的题，用动规解，记得可以推出一个规则的。
```python
class Solution(object):
    def uniquePaths(self, m, n):
        dp = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                dp[j] = dp[j - 1] + dp[j]
        return dp[-1] if m and n else 0
```
## 64. Minimum Path Sum
- 2019-04-21

这种全局最优显然要考虑递归，类似背包问题。
```python
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        c,r = len(grid),len(grid[0])    
        for i in range(c):
            for j in range(r):
                if i ==0 and j ==0:
                    continue
                elif i == 0:
                    grid[i][j] += grid[i][j-1]
                elif j == 0:
                    grid[i][j] += grid[i-1][j]
                else:
                    grid[i][j] += min(grid[i][j-1], grid[i-1][j])              
        return grid[-1][-1]
```
## 53. Maximum Subarray
- 2019-04-22

这题乍看上去需要循环尝试很多，看到一种很厉害的解法：
```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(1, len(nums)):
            if nums[i-1] > 0:
                nums[i] += nums[i-1]
        return max(nums)
```
## 74. Search a 2D Matrix

- 2019-04-23

简单搜索比较容易，O(log(n))解法还是挺难想的：
```
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or target is None:
            return False

        rows, cols = len(matrix), len(matrix[0])
        low, high = 0, rows * cols - 1
        
        while low <= high:
            mid = (low + high) / 2
            num = matrix[mid / cols][mid % cols]

            if num == target:
                return True
            elif num < target:
                low = mid + 1
            else:
                high = mid - 1
        
        return False
```
## 77. Combinations

- 2019-04-24

组合，实现C{n,m}，itertools里有轮子combinations
```
if k == 0:
            return [[]]
        return [pre + [i] for i in range(k, n+1) for pre in self.combine(i-1, k-1)]
```
可能更好理解些：
```
class Solution:
    def combine(self, n, k):
        stack = []
        res = []
        l, x = 0, 1
        while True:
            
            if l == k:
                res.append(stack[:])
            if l == k or n-x+1 < k-l:
                if not stack:
                    return res
                x = stack.pop() + 1
                l -= 1
            else:
                stack.append(x)
                x += 1
                l += 1
```
## 66. Plus One 
- 2019-04-25

easy题
```python
class Solution(object):
    def plusOne(self, digits):
        digits[-1] += 1
        for i in range(len(digits)-1, 0, -1):
            if digits[i] != 10:
                break
            digits[i] = 0
            digits[i-1] += 1
    
        if digits[0] == 10:
            digits[0] = 0
            digits.insert(0,1)
        return digits
```
## 78. Subsets
- 2019-04-26

求一个集合的所有子集，
```python
class Solution(object):
    def subsets(self, nums):
        res = [[]]
        for num in sorted(nums):
            res =res+ [item+[num] for item in res]
        return res
```
## 90. Subsets II
- 2019-04-29

和原先子集题（78）的区别是这个可以有重复元素；
```python
class Solution(object):
    def subsetsWithDup(self, nums):
        if not nums:
            return []
        nums.sort()
        res, cur = [[]], []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                cur = [item + [nums[i]] for item in cur]
            else:
                cur = [item + [nums[i]] for item in res]
            res += cur
        return res
```



## 80. Remove Duplicates from Sorted Array II
- 2019-04-27

要在O(1)空间内实现，就靠循环和if了；
```python
class Solution(object):
    def removeDuplicates(self, nums):
        i = 0
        for n in nums:
            if i < 2 or n > nums[i-2]:
                nums[i] = n
                i += 1
        return i
```
## 67. Add Binary
- 2019-04-28

用二进制的进位逻辑去算，不过输入输出都是字符串，不过就0，1两种字符，不需要转int了
```python
class Solution(object):
    def addBinary(self, a, b):#按逐个进位写的巨长的代码
        na=len(a)
        nb=len(b)
        res=[]
        rf={'0':0,'1':1}
        aone=0 #进位 add new one
        if na<nb:
            for i in range(na):
                r=rf[a[na-i-1]]+rf[b[nb-i-1]]+aone
                if r==3:
                    res.append('1')
                    aone=1
                elif r==2:
                    res.append('0')
                    aone=1
                elif r==1:
                    res.append('1') #0 or 1
                    aone=0 #?
                elif r==0:
                    res.append('0')
                    aone=0
                    
            for i in range(nb-na-1,-1,-1):
                r=rf[b[i]]+aone
                if r==2:
                    res.append('0') #aone=1
                elif r==1:
                    res.append('1')
                    aone=0
                elif r==0:
                    res.append('0')
                    aone=0
        else:#na>=nb
            for i in range(nb):
                r=rf[a[na-i-1]]+rf[b[nb-i-1]]+aone
                if r==3:
                    res.append('1')
                    aone=1
                elif r==2:
                    res.append('0')
                    aone=1
                elif r==1:
                    res.append('1') #0 or 1
                    aone=0 #?
                elif r==0:
                    res.append('0')
                    aone=0
            for i in range(na-nb-1,-1,-1):
                r=rf[a[i]]+aone
                if r==2:
                    res.append('0') #aone=1
                elif r==1:
                    res.append('1')
                    aone=0
                elif r==0:
                    res.append('0')
                    aone=0
        if aone==1:
            res.append('1')
        ors=[]
        for i in range(len(res),0,-1):
            ors.append(res[i-1])
        return ''.join(ors)
```
而看讨论区，一行代码有：`return bin(eval('0b' + a) + eval('0b' + b))[2:]`，0b means that the number that follows is in binary. `return f"{int(a,2)+int(b,2):b}"`；

## 91. Decode Ways
- 2019-04-30

有趣的题目，标个:star，之后再解。

```python
# todo
```
## 180. Consecutive Numbers
- 2019-05-01

五一第一天写个SQL题；
```
SELECT DISTINCT
    l1.Num AS ConsecutiveNums
FROM
    Logs l1,
    Logs l2,
    Logs l3
WHERE
    l1.Id = l2.Id - 1
    AND l2.Id = l3.Id - 1
    AND l1.Num = l2.Num
    AND l2.Num = l3.Num
;
```


## 37. 解数独
（现在上leetcode默认都跳中文官网了）

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        #x,y=(0,0) if self._b[0][0]==0 else self.getNext(0,0)
        if board[0][0]=='.':#更容易理解的写法
            self.trysxy(0,0,board)
        else:
            x,y=self.getNext(0,0,board)
            self.trysxy(x,y,board)
    def checkNotSame(self,x,y,val,board):#检查每行、每列及宫内是否有和b[x,y]相同项
        for row_item in board[x]: #第x行
            if row_item==val:
                return False
        for rows in board:#y所在列
            if rows[y]==val:
                return False
        ax=x//3*3 #把0~3中的值映射到[0,3]
        ab=y//3*3
        for r in range(ax,ax+3):
            for c in range(ab,ab+3):#注意r==x & c==y的情况下，其实没必要，val不会是0
                if board[r][c]==val:
                    return False
        return True
    def getNext(self,x,y,board): #得到下一个未填项,从x,y往下数，值等于0就返回新下标
        for ny in range(y+1,9): #下标是[0,8]
            if board[x][ny]=='.':
                return (x,ny)
        for row in range(x+1,9):
            for ny in range(0,9):
                if board[row][ny]=='.':
                    return (row,ny)
        return (-1,-1) #不存在下一个未填项的情况
    def getPrem(self,x,y,board): #得到x，y处可以填的值
        prem=[]
        rows=list(board[x])
        rows.extend([board[i][y] for i in range(9)])
        cols=set(rows)
        for i in range(1,10):
            i=str(i)
            if i not in cols:
                prem.append(i)
        return prem
    def trysxy(self,x,y,board): #主循环，尝试x，y处的解答
        if board[x][y]=='.': #不等于0的情况在调用外处理
            pv=self.getPrem(x,y,board)
            for v in pv:
                if self.checkNotSame(x,y,v,board):# 符合 行列宫均满足v符合条件 的
                    board[x][y]=v
                    nx,ny=self.getNext(x,y,board) #得到下一个0值格
                    if nx==-1: #没有下一个0格了；and ny==-1可以写但没必要
                        return True
                    else:
                        _end=self.trysxy(nx,ny,board) #向下尝试,递归
                        if not _end:
                            board[x][y]='.' #回溯，继续for v循环
                            #只需要改x，y处的值，不改其他值
                        else:
                            return True
```
回溯法，时间效率还行，因为是递归，空间耗费比较大，

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        if board[0][0]!='.':
            return self.check(0,0,board)
        else:
            nx,ny=self.getNext(0,0,board)
            if nx==-1:
                return True
            return self.check(nx,ny,board)
    def check(self,x,y,b):#检查数独是否合法
        v=b[x][y]
        for r in range(0,9):
            if r!=x:
                if b[r][y]==v:
                    return False
            if r!=y:
                if b[x][r]==v:
                    return False
        ax=x//3*3
        ab=y//3*3
        for r in range(ax,ax+3):
            for c in range(ab,ab+3):
                if b[r][c]==v and r!=x and c!=y:
                    return False
        nx,ny=self.getNext(x,y,b)
        if nx==-1:
            return True
        return self.check(nx,ny,b)
    def getNext(self,x,y,b):
        for ny in range(y+1,9):
            if b[x][ny]!='.':
                return (x,ny)
        for r in range(x+1,9):
            for ny in range(0,9):
                if b[r][ny]!='.':
                    return (r,ny)
        return (-1,-1)
```

## 42. Trapping Rain Water

这题2018年在面试中遇到过，当时想了好久才推出递归写法，早就后悔没多刷题了，看到这题更后悔没早点多刷题了。不愧是第42题。
```python
# 等我找到我以前的解法

```



逆序的好处：方便add；
顺序的好处：翻下来的时候可以顺路看之前的笔记，顺序符合心理认知。







