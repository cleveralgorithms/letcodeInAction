/**
 * 1) 单链表反转
 * 2) 链表中环的检测
 * 3) 两个有序的链表合并
 * 4) 删除链表倒数第n个结点
 * 5) 求链表的中间结点
 */

 class Node{
     constructor(element){
        this.element = element;
        this.next = null
    }
 }

 class Linkedlist{
     constructor(){
        this.head = new Node('head')
     }
    //
findByValue(item){
    let currentNode = this.head
    while(currentNode!==null&&currentNode.element!==item){
        currentNode = currentNode.next
    }
    return currentNode===null?-1:currentNode
}

// findbyIndex
findByIndex(index){
    let currentNode = this.head
    let pos = 0
    while(currentNode!==null&&pos!==index) {
        currentNode = currentNode.next
        pos++
    }
    return currentNode===null?-1:currentNode
}

//制动元素向后插入
findPrev(item){
    let currentNode =this.head
    while(currentNode.next!==null&&currentNode.next.element!==item){
        currentNode = currentNode.next
    }
    if (currentNode===null) {
        return-1
    }
    return currentNode
}

//根据值删除
remove(item){
    const desNode = this.findByValue(item)
    if(desNode ===-1){
        console.log('not find')
        return
    }
    const prevNode = this.findPrev(item)
    prevNode.next = desNode.next
    }
}
//
// 遍历所有节点
display(){
    // 先检查是否为有环
    if(this.checkCircle()) return false

    let currentNode = this.head
    while (currentNode!==null){
        console.log(currentNode.element)
        currentNode = currentNode.next
    }
}

// 翻转单列表  尾插法
reverseList() {
    const root = new Node('head')
    let currentNode = this.head.next
    while(currentNode!==null){
        const next  = currentNode.next
        currentNode.next = root.next
        root.next = currentNode
        currentNode = next  
    }
}
// reverse  就地翻转法
reverseList1() {
    let currentNode = this.head.next
    let previouNode = null
    while(currentNode!==null){
        let nextNode = currentNode.next
        currentNode.next = previouNode
        previouNode = currentNode
        currentNode = nextNode
    }
    this.head.next = previouNode
}




 }