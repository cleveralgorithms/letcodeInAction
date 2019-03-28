// 未完全实现

class Method {
  constructor(props, context) {
    this.sum = 8
    this.k = [2, 4]
    this.t = []
  }

//   判断函数
  judge(i) {
    if (this.add(this.t) == this.sum && this.k.includes(i)) {
      console.log(this.t)
    }
  }

  // 加 函数
  add(array) {
    return array.reduce((a, b) => a + b)
  }

//  计算函数
  calculate() {
    this.k.forEach((item, i) => {
      const diff = this.sum - this.add(this.k)
      // 第一种：单个元素
      if (this.sum % item == 0) {
        // 临时数组
        let tmpArray = []
        const lengthOfTmpArrray = this.sum / item
        for (let i = 0; i < lengthOfTmpArrray; i++) {
          tmpArray.push(item)
        }
        this.t.push(tmpArray)
      }
      // 如果能够直接找到匹配的元素  数据确定
      let secondTmpArray = this.k
      if (diff == item) {
        secondTmpArray.push(item)
        this.t.push(secondTmpArray)
      }
    })

  }

//  排列组合
  doExchange(arr, index) {
    let result = []
    arr.forEach((item, i) => {
      result[index] = arr[index][i]
      if (index != arr.length - 1) {
        this.doExchange(arr, index + 1)
      } else {
        result.push(result.join(','))
      }
    })
    return result
  }
}

const method = new Method()

method.calculate()
method.doExchange(method.t, 0)
console.log('t:', method.t)

