class Solution():
    def __init__(self):
        pass

    '''
    [Leetcode: Easy] (1) Two Sum I, 09/12/2015

    Given an array of integers, find two numbers such that they add up to a specific target number.
    The function twoSum should return indices of the two numbers such that they add up to the target,
    where index1 must be less than index2.
    Ex)
        Input: numbers={2, 7, 11, 15}, target=9
        Output: index1=1, index2=2
    '''
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        def order(a, b):
            if a > b:
                return [b,a]
            else:
                return [a,b]
        
        # Using hash table
        index = 1
        lookup = {}
        for num in nums:
            lookup[num] = index
            index += 1
        
        for n in nums:
            val = abs(target - n)
            try:
                if lookup[val] > 0:
                    if lookup[val] == n:
                        continue
                    else:
                        return order(lookup[n], lookup[val])
            except:
                pass
            
        '''
        # Simple hack
        index1 = 0
        for element in nums:
            inspection = nums[:]
            inspection.pop(index1)
            val = abs(target - element)
            if val in inspection:
                if val == nums[index1]:
                    index2 = inspection.index(val) + 2
                else:
                    index2 = nums.index(val) + 1
                return order(index1 + 1, index2)
            index1 += 1
        '''
        return None

    '''
    [Leetcode: Easy] (258) Add Digits

    Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
        Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2.
        Since 2 has only one digit, return it.
    '''
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        def addDigitsHelper(n):
            sum = 0
            for i in str(n):
                sum += int(i)
            return sum

        r = addDigitsHelper(num)

        while len(str(r)) > 1:
            r = addDigitsHelper(r)

        return r

    '''
    [Leetcode: Easy] (104) Maximum Depth of Binary Tree, 06/08/2015

    Given a binary tree, find its maximum depth.
    The maximum depth is the number of nodes along the longest path
    from the root node down to the farthest leaf node.

    # Definition for a binary tree node.
    # class TreeNode(object):
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None
    '''
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if root is None:
            return 0

        if root.left is None and root.right is None:
            return 1

        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    '''
    [Leetcode: Easy] (283) Move Zeroes

    Given an array nums, write a function to move all 0's to the end of it
    while maintaining the relative order of the non-zero elements.

        given nums = [0, 1, 0, 3, 12],
        after calling your function,
        nums should be [1, 3, 12, 0, 0].
    '''
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        i = 0
        zero_cnt = nums.count(0)

        for n in nums:
            if n == 0:
                pass
            else:
                nums[i] = n
                i += 1

        for x in range(zero_cnt):
            nums[i] = 0
            i += 1

    '''
    [Leetcode: Easy] (242) Valid Anagram

    Given two strings s and t, write a function to determine if t is an anagram of s.
    Ex)
        s = "anagram", t = "nagaram", return true.
        s = "rat", t = "car", return false.
    '''
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        def word_analysis(word):
            elements = dict()
            for w in [x for x in word]:
                if w not in elements:
                    elements[w] = 1
                else:
                    elements[w] += 1
            return elements

        if s == "" and t == "":
            return True

        return word_analysis(s) == word_analysis(t)

    def pick_largest(self, n):
        import math
        # the Largest number to check
        limit = int(math.sqrt(n))
        check = [x*x for x in range(1,limit+1)]
        print list(reversed(range(len(check)-1)))
        for i in list(reversed(range(len(check)-1))):
            print i
            square = limit*limit
            if check[i] <= n - square < check[i+1]:
                return check[i], n-square

    # Reverse the row of the matrix
    def performOps(self, A):
        m = len(A)
        n = len(A[0])
        B = []
        for i in xrange(len(A)):
            B.append([0] * n)
            for j in xrange(len(A[i])):
                B[i][n - 1 - j] = A[i][j]
        return B

    '''
    [Leetcode: Easy] (263) Ugly Number, 12/08/2015

    Write a program to check whether a given number is an ugly number.
    Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.
    For example, 6, 8 are ugly while 14 is not ugly since it includes another prime factor 7.
    Note that 1 is typically treated as an ugly number.
    '''
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        def isUglyHelper(n):
           if n % 2 == 0:
               return n // 2
           elif n % 3 == 0:
               return n // 3
           elif n % 5 == 0:
               return n // 5
           else:
               return n

        if num <= 0:
            return False

        if num == 1:
            return True

        while num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
            if num in [2,3,5]:
                break
            num = isUglyHelper(num)

        if num > 5:
            return False

        return True

    '''
    [Leetcode: Easy] (217) Contains Duplicates I, 12/11/2015

    Given an array of integers, find if the array contains any duplicates.
    Your function should return true if any value appears at least twice in the array,
    and it should return false if every element is distinct.
    '''
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return False if len(nums) == 0 else len(set(nums)) != len(nums)

    '''
    [Leetcode: Easy] (219) Contains Duplicates II, 12/11/2015

    Given an array of integers and an integer k,
    find out whether there are two distinct indices i and j in the array
    such that nums[i] = nums[j] and the difference between i and j is at most k.
    '''
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        counter = {}
        ans = False

        for i in range(len(nums)):
            if nums[i] not in counter.keys():
                counter[nums[i]] = [i]
            else:
                counter[nums[i]] += [i]

        for positions in counter.values():
            if len(positions) > 1:
                for gap in [positions[x+1] - positions[x] for x in range(len(positions) - 1)]:
                    if gap <= k:
                        ans = True

        return ans

    '''
    [Leetcode: Easy] (237) Delete Node in a Linked List, 12/17/2015

    Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.
    Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node with value 3,
    the linked list should become 1 -> 2 -> 4 after calling your function.
    '''
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        # Elegant O(1) solution,
        # 1. Copy the next value to the current node to delete
        # 2. Link the next node of the current node to the next node of the copied node
        node.val = node.next.val
        node.next = node.next.next

    '''
    [Leetcode: Easy] (110) Balanced Binary Tree, 12/19/2015

    Given a binary tree, determine if it is height-balanced.
    For this problem, a height-balanced binary tree is defined as a binary tree
    in which the depth of the two subtrees of every node never differ by more than 1.
    '''
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def isBalancedHelper(root):
            if root is None:
                return 0

            left = isBalancedHelper(root.left)
            right = isBalancedHelper(root.right)

            # Check if each subtree is balanced on both left and right
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1

            return 1 + max(left, right)

        return False if isBalancedHelper(root) == -1 else True