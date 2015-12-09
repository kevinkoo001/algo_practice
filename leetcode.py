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