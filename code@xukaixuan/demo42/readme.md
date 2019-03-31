## 接雨水

### 考察知识点：

栈，双指针

### 题目描述：

![示意图](rainwatertrap.png)
```
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.



The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped. Thanks Marcos for contributing this image!

Example:

Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```
### 解题思路
1. 遍历两次```height```数组，并且得到```height[i]```元素左、右最高值（wall）

2. 再次遍历数组，将```height[i]```与其左右临近最高边界的最小的’wall‘ 比较，计算```total trapping-rain-water```