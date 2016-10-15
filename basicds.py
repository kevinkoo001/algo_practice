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

'''
[Leetcode: Easy] (232) Implement Queue using Stacks, 6/11/2016
Implement the following operations of a queue using stacks.

push(x) -- Push element x to the back of queue.
pop() -- Removes the element from in front of queue.
peek() -- Get the front element.
empty() -- Return whether the queue is empty.

Notes:
You must use only standard operations of a stack --
which means only push to top, peek/pop from top, size, and is empty operations are valid.
Depending on your language, stack may not be supported natively.
You may simulate a stack by using a list or deque (double-ended queue),
as long as you use only standard operations of a stack.
You may assume that all operations are valid
(for example, no pop or peek operations will be called on an empty queue).
'''
class Queue(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        # We need three stacks to mimic a queue
        self.stack1 = []
        self.stack2 = []
        self.tmp_stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """

        # When a queue is empty, start to store data into stack1
        if self.empty():
            self.stack1.append(x)

        # If the first stack is empty
        #  a) store data into stack2,
        #  b) pop off all data in stack1 to tmp_stack
        #  c) pop off all data in tmp_stack to stack2
        elif len(self.stack1) > 0:
            self.stack2.append(x)
            while len(self.stack1) > 0:
                self.tmp_stack.append(self.stack1.pop(-1))
            while len(self.tmp_stack) > 0:
                self.stack2.append(self.tmp_stack.pop(-1))
            self.stack1 = []

        # If the second stack is empty, do the same in a reverse way
        # Popping off all data twice can be eventually FIFO structure from LIFO
        elif len(self.stack2) > 0:
            self.stack1.append(x)
            while len(self.stack2) > 0:
                self.tmp_stack.append(self.stack2.pop(-1))
            while len(self.tmp_stack) > 0:
                self.stack1.append(self.tmp_stack.pop(-1))
            self.stack2 = []

    def pop(self):
        """
        :rtype: nothing
        """
        if self.empty():
            return None

        if len(self.stack1) > 0:
            self.stack1.pop(-1)

        if len(self.stack2) > 0:
            self.stack2.pop(-1)

    def peek(self):
        """
        :rtype: int
        """
        if self.empty():
            return None

        if len(self.stack1) > 0:
            return self.stack1[-1]

        if len(self.stack2) > 0:
            return self.stack2[-1]

    def empty(self):
        """
        :rtype: bool
        """
        return len(self.stack1) == 0 and len(self.stack2) == 0

'''
[Leetcode: Medium] (398) Random Pick Index, 10/15/2016
Given an array of integers with possible duplicates, randomly output the index of a given target number.
You can assume that the given target number must exist in the array.

Note:
The array size can be very large. Solution that uses too much extra space will not pass the judge.

Example:
int[] nums = new int[] {1,2,3,3,3};
Solution solution = new Solution(nums);

// pick(3) should return either index 2, 3, or 4 randomly.
Each index should have equal probability of returning.
solution.pick(3);

// pick(1) should return 0. Since in the array only nums[0] is equal to 1.
solution.pick(1);
'''
class RandomPickIndex(object):
    def __init__(self, nums):
        """
        :type nums: List[int]
        :type numsSize: int
        """

        self.info_nums = {}

        # Construct the hash table for all pairs
        # element: the list of indexes that the element is shown
        for i, n in enumerate(nums):
            if n not in self.info_nums:
                self.info_nums[n] = [i]
            else:
                self.info_nums[n].append(i)

    def pick(self, target):
        """
        :type target: int
        :rtype: int
        """
        indexes = self.info_nums[target]

        if len(indexes) == 1:
            return indexes[0]

        import random
        random.shuffle(indexes)

        # Pick the one of the indexes for the corresponding target
        return indexes[0]