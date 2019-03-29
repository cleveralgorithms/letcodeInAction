/*
 * @lc app=leetcode id=2 lang=javascript
 *
 * [2] Add Two Numbers
 *
 * https://leetcode.com/problems/add-two-numbers/description/
 *
 * algorithms
 * Medium (30.80%)
 * Total Accepted:    807.8K
 * Total Submissions: 2.6M
 * Testcase Example:  '[2,4,3]\n[5,6,4]'
 *
 * You are given two non-empty linked lists representing two non-negative
 * integers. The digits are stored in reverse order and each of their nodes
 * contain a single digit. Add the two numbers and return it as a linked list.
 * 
 * You may assume the two numbers do not contain any leading zero, except the
 * number 0 itself.
 * 
 * Example:
 * 
 * 
 * Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
 * Output: 7 -> 0 -> 8
 * Explanation: 342 + 465 = 807.
 * 
 * 
 */
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */
var addTwoNumbers = function(l1, l2) {
    var add =0, ans,head;
    while(l1||l2){
        var a = l1?l1.val:0,
        b = l2?l2.val:0;
        var sum = a+b+add;
        add = ~~(sum/10);
        var node = new ListNode(sum%10);
        if(!ans)ans = head = node;
        else{
            head.next = node;
            head = node;
        }
        if(l1){
            l1 = l1.next;
        }
        if(l2){
            l2 = l2.next;
        }
        }
        if(add){
            var node = new ListNode(add);
            head.next = node;
            head = node;
    }
    return ans;
    
};

