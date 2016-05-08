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