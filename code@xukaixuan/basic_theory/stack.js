/**
 * 基于链表实现的栈
 */

 class Node{
     constructor(element){
         this.element = element
         this.next  = null
     }
 }

 class StackBasedLinkedList{
     constructor(){
         this.top =null
     }
     push(value){
         const node = new Node(value)
         if(this.top ===null){
             this.top = node
         }else{
             node.next = this.top
             this.top = node
         }
     }
     pop(){
         if (this.top ===null){
             return -1
         } 
         const value = this.top.element
         this.top = this.top.next
         return value
     }
     // 实现浏览器前进后退功能
     clear(){
         this.top = null
     }
     display(){
         if(this.top!==null){
             let temp = this.top
             while(temp!==null){
                 console.log(temp.element)
                 temp = temp.next
             }
         }
     }
 }
 // Test
 const newStack = new StackBasedLinkedList()
 newStack.push(1)
 newStack.push(2)
 newStack.push(3)
 newStack.push(4)
 newStack.pop()
 