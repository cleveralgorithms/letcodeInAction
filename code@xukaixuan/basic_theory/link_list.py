#link_list

class Element(object):
    def __init__(self,value):
        self.value =value
        self.next = None
class LinkedList(object):
    def __init__(self,head =None):
        self.head = head

    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
        else: self.head =new_element
    
    def get_position(self, position):
        """Get an element from a particular position.
        Assume the first position is "1".
        Return "None" if position is not in the list."""
       counter = 1
       current = self..head
       if position < 1:
           return None
        while  currtrnt and counter < position:
            if counter == position:
                return current
            counter+=1
            current = current.next
        return None

    def insert(self,new_element,position):
        counter = 1
        current = self.head
        if position > 1:
            while current and counter < position:
                if counter== position-1:
                    current.next = new_element
                    new_element.next = current.next
                current,counter = current.next,counter+1
        elif position ==1:
            new_element.next = self.head
            self.head = new_element
    def delete(self,value):
        current = self.head
        pre = None
        while current.value !=value and current.next:
            pre,current = current,current.next
        if pre.value == value:
                if pre:
                    pre.next = current.next
                else:
                    self.head = current

        
