class ListNode(object):
    def __init__(self, x, next = None):
        self.val = x
        self.next = next

class TreeNode(object):
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right

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