


### 2019-03-21
今晚刷了网易的一道题，输入n, t, a，返回最高分，开始觉得很简单，但后来分析觉得最高分是一个最优化问题，通过集合的交集关系分析觉得最值应该在两个端点，
也就是t完全被a包括或者a完全被t包括；所以就直接代入两个端点的情况了，代码能通过所有的测试用例：
```python
lst=list(map(int, input().split()))
n,t,a=lst[0],lst[1],lst[2]
if t<a: #t小于a时， t完全被a包括
    print(t+n-a)
else:
    print(a+n-t)

```

### 2019-03-30
快手2019春招算法A卷。

判断序列是否是非递减序列，当不是的时候，可以改变一个数字，再判断。返回最终的判断结果。是则输出1，否则0。
测试用例：`[3,4,6,5,5,7,8]`，output：`1`，本身不是，但把第3位的6变成4或5就是了，所以输出 1 。

```python
lst=list(map(int,input().split()))

isUsing=False #是否使用了一次换数字的机会
flag=1 #最终输出结果
for i in range(len(lst)-1):
    if lst[i] <= lst[i+1]: #判断时取第i位和第i+1位比较，所以选i in [0,len()-2]
        pass #往下判断
    else:
        if isUsing:#用过了一次机会
            flag=0 
            break #因为nowcoder是print输出所以不用return
        else:
            k=lst[i+1]
            if i==0: #防止i-1溢出
                isUsing=True
                lst[i]=k
            elif lst[i-1]<=k:#进行一次替换
                isUsing=True
                lst[i]=k
            else:#替换也没有用了
                flag=0
                break
print(flag)
```




