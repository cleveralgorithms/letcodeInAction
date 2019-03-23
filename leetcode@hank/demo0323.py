# 使用python实现10进制数转2进制数，位移操作及与运算,未考虑负数

def int2binary(num):
    result = []
    while num != 0:
        result.append(num & 1)
        num = num >> 1
    result.reverse()
    return result

print(int2binary(10))