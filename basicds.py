class ListNode(object):
    def __init__(self, x, next = None):
        self.val = x
        self.next = next

class TreeNode(object):
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right

class TreeLinkNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None
         self.next = None

class Interval(object):
     def __init__(self, s=0, e=0):
         self.start = s
         self.end = e

# Code from http://rosettacode.org/wiki/Tree_traversal
def levelorder(node, more=None):
    if node is not None:
        if more is None:
            more = []
        more += [node.left, node.right]
        print node.val,
    if more:
        levelorder(more[0], more[1:])

def printList(root):
    if root is None:
        print 'No entry in the linked list!'
    while root:
        print root.val,
        root = root.next
    print ''

# Definition for a point (for leetcode 149)
class Point(object):
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b

'''
[Leetcode: Easy] (155) Min Stack, 6/8/2016
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.
'''
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.minval = 2**32 # INT_MAX

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stack.append(x)

        if self.minval > x:
            self.minval = x

    def pop(self):
        """
        :rtype: void
        """
        val = self.stack.pop()

        # Initialization when no element left
        if len(self.stack) == 0:
            self.minval = 2**32

        # Adjust min value when min is popped out
        elif val == self.minval:
            self.minval = min(self.stack)

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.minval