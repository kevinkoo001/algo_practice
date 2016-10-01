# coding=utf-8
import basicds

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
<<<<<<< HEAD
    [Leetcode: Easy] (226) Invert Binary Tree, 12/17/2015
         4
       /   \
      2     7
     / \   / \
    1   3 6   9
        to
         4
       /   \
      7     2
     / \   / \
    9   6 3   1
    '''
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root is None:
            return None

        if root.left:
            self.invertTree(root.left)
        if root.right:
            self.invertTree(root.right)

        root.left, root.right = root.right, root.left
        return root

    '''
    [Leetcode: Easy] (234) Palindrome Linked List, 1/30/2016

    Given a singly linked list, determine if it is a palindrome.
    Could you do it in O(n) time and O(1) space? Yes!
    '''
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """

        def getSize(head):
            if head is None:
                return 0
            else:
                size = 1
                ptr = head
                while ptr.next:
                    size += 1
                    ptr = ptr.next
                return size

        panLen = getSize(head)
        if panLen < 2:
            return True
        elif panLen == 2:
            return head.val == head.next.val
        elif panLen == 3:
            return head.val == head.next.next.val

        # Initialization
        # head --|
        #        O <- O   O -> ...
        # left_head --|   |-- right_head
        left_head = head.next
        right_head = left_head.next
        left_head.next = head

        # Reverse the link up to the half of the linked list
        # head --|
        #        O <- O <- O   O -> ...
        # left_head -------|   |-- right_head
        for i in range(1, panLen//2 - 1):
            p_tmp = left_head
            left_head = right_head
            right_head = right_head.next
            left_head.next = p_tmp

        if panLen % 2 == 1:
            right_head = right_head.next

        while left_head is not None and right_head is not None:
            if left_head.val != right_head.val:
                return False
            left_head = left_head.next
            right_head = right_head.next

        return True

    '''
    [Leetcode: Medium] (49) Group Anagrams, 1/30/2016

    Given an array of strings, group anagrams together.
    For the return value, each inner list's elements must follow the lexicographic order.
    All inputs will be in lower-case

    For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
    Return:

    [
      ["ate", "eat","tea"],
      ["nat","tan"],
      ["bat"]
    ]
    '''
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """

        # The below should be working, but somehow time-exceeded constraint frustrates me
        ans = dict()

        for s in strs:
            group = hash(''.join(sorted(s)))
            if group in ans.keys():
                ans[group] += [s]
            else:
                ans[group] = [s]

        result = []
        for v in ans.values():
            if len(v) > 1:
                result.append(sorted(v))
            else:
                result.append(v)

        '''
        # Simple but ugly using a library
        from itertools import groupby
        return [sorted(list(group)) for key, group in groupby(sorted(strs, key=sorted),sorted)]
        '''

        return result

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

    '''
    [Leetcode: Medium] (304) Range Sum Query (Immutable), 1/30/2016

    Given a 2D matrix matrix, find the sum of the elements inside the rectangle
    defined by its upper left corner (row1, col1) and lower right corner (row2, col2)
    Given matrix = [
      [3, 0, 1, 4, 2],
      [5, 6, 3, 2, 1],
      [1, 2, 0, 1, 5],
      [4, 1, 0, 1, 7],
      [1, 0, 3, 0, 5]
    ]
    sumRegion(2, 1, 4, 3) -> 8
    sumRegion(1, 1, 2, 2) -> 11
    sumRegion(1, 2, 2, 4) -> 12
    You may assume that the matrix does not change.
    There are many calls to sumRegion function.
    You may assume that row1 <= row2 and col1 <= col2.
    '''
    class NumMatrix(object):
        def __init__(self, matrix):
            """
            initialize your data structure here.
            :type matrix: List[List[int]]
            """

            self.matrix = matrix

            # dp matrix initialization
            self.dp = [[0 for col in range(len(matrix[0]))] for row in range(len(matrix))]

            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    upper_left = self.dp[i-1][j-1] if i-1 >= 0 and j-1 >= 0 else 0
                    left = self.dp[i-1][j] if i-1 >= 0 else 0
                    up = self.dp[i][j-1] if j-1 >= 0 else 0
                    self.dp[i][j] = up + left - upper_left + matrix[i][j]

        def sumRegion(self, row1, col1, row2, col2):
            """
            sum of elements matrix[(row1,col1)..(row2,col2)], inclusive.
            :type row1: int
            :type col1: int
            :type row2: int
            :type col2: int
            :rtype: int
            """

            # Using dp matrix, the sum of the region can be calculated as
            # dp[row2][col2] - dp[row2][col1-1] - dp[row1-1][col2] + dp[row1-1][col1-1]
            sum_upper_left = self.dp[row1-1][col1-1] if row1-1 >= 0 and col1-1 >=0 else 0
            sum_left = self.dp[row2][col1-1] if col1-1 >= 0 else 0
            sum_up = self.dp[row1-1][col2] if row1-1 >=0 else 0

            #print sum_upper_left, sum_left, sum_up, dp[row2][col2]
            return self.dp[row2][col2] - (sum_left + sum_up) + sum_upper_left

    '''
    [Leetcode: Easy] (292) Nin game, 1/31/2016

    You are playing the following Nim Game with your friend:
    There is a heap of stones on the table, each time one of you take turns to remove 1 to 3 stones.
    The one who removes the last stone will be the winner. You will take the first turn to remove the stones.
    Both of you are very clever and have optimal strategies for the game.
    Write a function to determine whether you can win the game given the number of stones in the heap.
    For example, if there are 4 stones in the heap, then you will never win the game:
    no matter 1, 2, or 3 stones you remove, the last stone will always be removed by your friend.
    '''
    def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """
        # Assuming optimal strategies would be taken at all times for both players,
        # the first player would always lose when n is the multiple of 4
        # because the other player will pick some so that the remaining stone is the multiple of 4.
        return n%4 != 0

    '''
    [Leetcode: Easy] (328) Odd Even Linked List, 1/31/2016

    Given a singly linked list, group all odd nodes together followed by the even nodes.
    You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.
    Example:
        Given 1->2->3->4->5->NULL,
        return 1->3->5->2->4->NULL.
    Note:
        We are talking about the node number and not the value in the nodes.
        The relative order inside both the even and odd groups should remain as it was in the input.
        The first node is considered odd, the second node even and so on ...
    '''
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        # No change if the number of nodes in the list is less than 3
        if head is None or head.next is None or head.next.next is None:
            return head

        # (Init) Define odd/even head ptrs and next ptrs
        # ( 1 )  -->  ( 2 )  -->  ( 3 )  -->  ( 4 )
        #  ||           ||
        #  ||-->head    ||-->even_head
        #  |-->odd_next |-->even_next

        even_head = head.next
        odd_next = head
        even_next = even_head

        # Step [A] odd_next.next = even_next.next
        # ( 1 )  -------------->  ( 3 ) --->  ( 4 )
        #              ( 2 )  ------|
        #  ||           ||
        #  ||-->head    ||-->even_head
        #  |-->odd_next |-->even_next

        # Step [B] odd_next = even_next.next
        # ( 1 )  -------------->  ( 3 ) --->  ( 4 )
        #              ( 2 )  ------||
        #   |           ||           |-->odd_next
        #   |-->head    ||-->even_head
        #               |-->even_next

        # Step [C] even_next.next = odd_next.next
        # ( 1 )  -------------->  ( 3 ) --------|
        #              ( 2 )  -------|----->   ( 4 )
        #   |           ||           |-->odd_next
        #   |-->head    ||-->even_head
        #               |-->even_next

        # Step [D] even_next = odd_next.next
        # ( 1 )  -------------->  ( 3 ) --------|
        #              ( 2 )  -------|----->   ( 4 )
        #   |            |           |-->odd_next|
        #   |-->head     |-->even_head           |
        #                                        |-->even_next

        # Step [E] Complete and return head
        # ( 1 )  -->  ( 3 )  -->  ( 2 )  -->  ( 4 )
        #   |            |           |-->even_head
        #   |-->head     |-->odd_next            |
        #                                        |-->even_next

        while odd_next is not None or even_next is not None:
            # When the number of linked list is even
            if even_next.next is None:
                odd_next.next = even_head           # Step [E]
                even_next.next = None
                break
            else:
                odd_next.next = even_next.next      # Step [A]
                odd_next = even_next.next           # Step [B]

            # When the number of linked list is odd
            if odd_next.next is None:
                odd_next.next = even_head
                even_next.next = None
                break
            else:
                even_next.next = odd_next.next      # Step [C]
                even_next = odd_next.next           # Step [D]

        return head

    '''
    [Leetcode: Easy] (206) Reverse Linked List, 2/3/2016

    Reverse a singly linked list.
    Example:
        Given 1->2->3->4->5->NULL,
        return 5->4->3->2->1->NULL.
    '''
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        if head is None or head.next is None:
            return head

        # Initialization
        prev = head
        cur = head.next
        head.next = None

        while cur.next:
            post = cur.next
            cur.next = prev
            prev = cur
            cur = post

        cur.next = prev

        return cur

    '''
    [Leetcode: Easy] (231) Power of Two, 2/4/2016

    Given an integer, write a function to determine if it is a power of two.
    '''
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n == 0:
            return False

        '''
        # Find the smallest power that satisfies n is larger than 2**power
        # The only case that n is power of two is n == 2 ** power
        # For example, n = 15; it ends up with finding power = 3,
        # 15 != 2**3: thus return False

        power = 0

        while n > (2**power):
            power += 1

        return n == 2**power
        '''

        # The simplest way is to check (n) & (n - 1) == 0
        # If n is the power of two, n can be represented as (100...0),
        # n-1 would be (011...1) accordingly. Thus (n & n-1) should be always zero
        return n & (n-1) == 0

    '''
    [Leetcode: Easy] (326) Power of Three, 2/4/2016

    Given an integer, write a function to determine if it is a power of three.
    (Followup) Could you do it without using any loop / recursion?
    '''
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n == 0:
            return False

        while n % 3 == 0:
            n = n // 3

        return n == 1

    '''
    [Leetcode: Easy] (190) Reverse Bits, 2/4/2016

    Reverse bits of a given 32 bits unsigned integer.
    For example,
    given input 43261596 (represented in binary as 00000010100101000001111010011100),
    return 964176192 (represented in binary as 00111001011110000010100101000000).

    (Follow-up) If this function is called many times, how would you optimize it?
    '''
    def reverseBits(self, n):
        """
        :type n: int
        :rtype: int
        """

        # Represent a given binary with 32 digits
        b = bin(n)[2:].zfill(32)

        ans = '0b'
        for i in range(len(b)-1, -1, -1):
            ans += b[i]

        return int(ans, 2)

    '''
    [Leetcode: Easy] (7) Reverse Integer, 2/4/2016
    Reverse digits of an integer. Consider the sign and INT_MIX/INT_MAX

    Example1: x = 123, return 321
    Example2: x = -123, return -321
    '''
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """

        # Define the range of signed integer expression
        INT_MAX = 2**31 - 1
        INT_MIN = -1 * (2**31)

        if x == 0 or x > INT_MAX:
            return 0

        v = str(abs(x))

        # Reverse the string then examine the range when returning
        ans = ''
        for i in range(len(v)-1, -1, -1):
            ans += v[i]

        ans = int(ans) if x>0 else (-1 * int(ans))

        return ans if INT_MIN <= ans <= INT_MAX else 0

    '''
    [Leetcode: Easy] (100) Same Tree, 2/4/2016
    Given two binary trees, write a function to check if they are equal or not.
    Two binary trees are considered equal if they are structurally identical
    and the nodes have the same value.
    '''
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """

        # Recursively visit all nodes of both p and q in order
        def isSameTreeHelper(p, q):
            # Base case 1: two nodes are None, return True
            if p is None and q is None:
                return True

            # Base case 2: one node is None and the other is not, return False
            elif (p is None and q is not None) or (p is not None and q is None):
                print "Topology is Different!"
                return False

            # Otherwise (two nodes are not None), check if the values are the same
            # Check out all subtrees of two trees: p and q.
            else:
                if p.val != q.val:
                    print "Different value here: %d in p and %d in q" % (p.val, q.val)
                return p.val == q.val and isSameTreeHelper(p.left, q.left) and isSameTreeHelper(p.right, q.right)

        return isSameTreeHelper(p, q)

    '''
    [Leetcode: Easy] (9) Palindrome Number, 2/6/2016
    Determine whether an integer is a palindrome. Do this without extra space.
    '''
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """

        if x < 0:
            return False
        elif x / 10 == 0:
            return True

        digit = len(str(abs(x)))//2

        if len(str(x)) % 2 != 0:
            return str(x // (10**(digit+1))) == str(x % (10**digit)).zfill(digit)[::-1]
        else:
            return str(x // (10**(digit))) == str(x % (10**(digit))).zfill(digit)[::-1]


    '''
    [Leetcode: Hard] (140) Word Break II, 2/6/2016

    Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where each word is a valid dictionary word.

    Return all such possible sentences.
    For example, given
    s = "catsanddog",
    dict = ["cat", "cats", "and", "sand", "dog"].
    A solution is ["cats and dog", "cat sand dog"].
    '''
    def wordBreak(self, s, wordDict):
        # THIS NEEDS TO BE THOUGHT (Not My Solution!)
        memo = {len(s): ['']}
        def sentences(i):
            if i not in memo:
                memo[i] = [s[i:j] + (tail and ' ' + tail)
                           for j in range(i+1, len(s)+1)
                           if s[i:j] in wordDict
                           for tail in sentences(j)]
                print memo
            return memo[i]
        return sentences(0)

    '''
    [Leetcode: Hard] (57) Insert Interval, 2/6/2016
    Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
    You may assume that the intervals were initially sorted according to their start times.

    Example 1:
    Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].
    Example 2:
    Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].
    This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].
    '''
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """

        # The following code has a bug to fix. (NOT PASSED ALL CASES YET)
        num_ranges = set()
        nums = set()
        newIntervals = []
        dots = {}

        # Suppose [s,e] represents s <= integers < e in num_ranges
        for i in intervals:
            for j in range(i.start, i.end, 1):
                num_ranges.add(j)

            # nums contains all edges to check special cases
            for k in range(i.start, i.end+1, 1):
                if not i.start == i.end:
                    nums.add(k)

            # Special case to handle in case of a dot! (i.e., (0,0), (1,1), ...)
            if i.start == i.end and i.start not in nums:
                dots[i.start] = i

        # nums set() holds all integers in the given intervals including new newInterval
        for new in range(newInterval.start, newInterval.end, 1):
            num_ranges.add(new)

        if newInterval.start == newInterval.end:
            num_ranges.add(newInterval.start)

        if newInterval.start == newInterval.end and newInterval.start not in nums:
            dots[newInterval.start] = newInterval

        num_ranges = sorted(num_ranges)
        range_start = None

        # By looking if adjacent number is consecutive, construct the order
        for (i, j) in [(num_ranges[k], num_ranges[k+1]) for k in range(len(num_ranges) - 1)]:
            if len(dots.keys()) > 0:
                dot = sorted(dots.keys())[0]
                if i > dot:
                    newIntervals.append(dots[dot])
                    dots.pop(dot)

            if range_start == None:
                range_start = i
                v = basicds.Interval(s=i)
                newIntervals.append(v)
            if j - i > 1:
                v.end = i+1
                range_start = None
            else:
                pass


        # Adjust the last range
        if num_ranges[-1] == newInterval.start:
            v.end = i+1
            newIntervals.append(basicds.Interval(j, j))

        elif j -i >= 2:
            newIntervals.append(basicds.Interval(j, j+1))
        else:
            v.end = j+1

        return newIntervals

    '''
    [Leetcode: Easy] (111) Minimum Depth of Binary Tree, 2/12/2016
    Given a binary tree, find its minimum depth.
    The minimum depth is the number of nodes along the shortest path
    from the root node down to the nearest leaf node.
    '''
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0

        elif root.left is None:
            return 1 + self.minDepth(root.right)

        elif root.right is None:
            return 1 + self.minDepth(root.left)

        else:
            return 1 + min(self.minDepth(root.left), self.minDepth(root.right))

    '''
    [Leetcode: Easy] (101) Symmetric Tree, 2/12/2016
    Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
    For example, this binary tree is symmetric:
         1
       / \
      2   2
     / \ / \
    3  4 4  3
    '''
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def isSymmetricHelper(left, right):
            # Both should be a leaf node
            if left is None and right is None:
                return True

            # The structure of the subtree does not match each other, it is asymmetric
            elif left is not None and right is None:
                return False
            elif left is None and right is not None:
                return False

            # When both nodes are end leaves, check the value. Also check the subtrees
            # if (left.right == right.left) AND (left.left == right.right) recursively!
            else:
                return left.val == right.val and \
                        isSymmetricHelper(left.right, right.left) and \
                        isSymmetricHelper(left.left, right.right)

        if root is None:
            return True

        return isSymmetricHelper(root.left, root.right)

    '''
    [Leetcode: Easy] (205) Isomorphic Strings, 2/13/2016
    Given two strings s and t, determine if they are isomorphic.
    Two strings are isomorphic if the characters in s can be replaced to get t.
    All occurrences of a character must be replaced with another character while preserving the order of characters.
    No two characters may map to the same character but a character may map to itself.
    '''
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        if len(s) == 0 and len(t) == 0:
            return True

        mapper = {}

        for i in range(len(s)):
            if s[i] not in mapper:
                if t[i] in mapper.values():
                    return False
                else:
                    mapper[s[i]] = t[i]
            else:
                if mapper[s[i]] == t[i]:
                    pass
                else:
                    return False

        return True

    '''
    [Leetcode: Medium] (331) Verify Preorder Serialization of a Binary Tree, 2/13/2016

        _9_
        /   \
       3     2
      / \   / \
     4   1  #  6
    / \ / \   / \
    # # # #   # #
    For example, the above binary tree can be serialized to the string "9,3,4,#,#,1,#,#,2,#,6,#,#", where # represents a null node.
    Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of a binary tree. Find an algorithm without reconstructing the tree.
    Each comma separated value in the string must be either an integer or a character '#' representing null pointer.
    You may assume that the input format is always valid, for example it could never contain two consecutive commas such as "1,,3".
    '''
    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """

        nodes = preorder.split(',')
        stack = []

        # If there is a single leaf, it is True
        if preorder == '#':
            return True
        # If any consecutive leaves has been found at first, it is False
        elif nodes[0] == '#' and nodes[1] == '#':
            return False

        # Putting the node to the stack until '#' is found;
        # If two consecutive sharps (##) are found, remove them with a previous node
        # until the condition is met any longer (while stmt)
        for n in range(len(nodes)):
            if len(stack) > 1 and nodes[n] == '#':
                if stack[-1] == '#':
                    while len(stack) > 1 and stack[-1] == '#':
                        stack.pop()
                        stack.pop()

            stack.append(nodes[n])

        # Finally only a single '#' remains intact in stack, which means a valid pre-ordered list
        return len(stack) == 1 and stack[0] == '#'

    '''
    [Leetcode: Medium] (131) Palindrome Partitioning, 2/13/2016
    Given a string s, partition s such that every substring of the partition is a palindrome.
    Return all possible palindrome partitioning of s.
    For example, given s = "aab", Return
    [
        ["aa","b"],
        ["a","a","b"]
    ]
    '''
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        def isPalindrome(s):
            if len(s) == 1:
                return True
            if len(s) == 2:
                return s[0] == s[1]

            return s[0] == s[-1] and isPalindrome(s[1:-1])

        # Need to look into how simple code just works more!
        def dfs(s, palindrome, ans):
            if len(s) == 0:
                ans.append(palindrome)
            for i in range(1, len(s)+1):
                if isPalindrome(s[:i]):
                    dfs(s[i:], palindrome+[s[:i]], ans)

        ans = []
        dfs(s, [], ans)
        return ans

    '''
    [Leetcode: Easy] (83) Palindrome Partitioning, 2/18/2016
    Given a sorted linked list, delete all duplicates such that each element appear only once.
    For example,
    Given 1->1->2, return 1->2.
    Given 1->1->2->3->3, return 1->2->3.
    '''
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return None

        # Define two pointers: prev and cur
        prev = head
        cur = prev.next

        while cur is not None:
            # As the linked list is sorted, if the value of prev and cur is the same;
            # Remove the link and update two pointers
            if prev.val == prev.next.val:
                prev.next = cur.next
                cur = cur.next

            # Otherwise just update the pointers, pointing next node
            else:
                prev = cur
                cur = cur.next

        return head

    '''
    [Leetcode: Easy] (27) Remove Element, 2/18/2016
    Given an array and a value, remove all instances of that value in place and return the new length.
    The order of elements can be changed. It doesn't matter what you leave beyond the new length.
    '''
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        for i in range(nums.count(val)):
            nums.remove(val)

        return len(nums)

    '''
    [Leetcode: Easy] (27) Remove Element, 2/20/2016
    Count the number of prime numbers less than a non-negative number, n.
    '''
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """

        '''
        # Naive way: O(n^2)
        def isPrime(x, primes):
            for j in [x for x in primes]:
                if i % j == 0:
                    return False
                    pass
            return True

        primes = [2]

        for i in range(2, n+1):
            if isPrime(i, primes) and i not in primes:
                primes.append(i)

        return len(primes) if n > 1 else 0
        '''

        if n <= 1:
            return 0

        # Initialize the sieve
        # [0, 1, 2, 3, 4, 5, 6, ...]
        # [F, F, T, T, T, T, T, ...]
        prime = [False, False] + [True]*(n-2)

        x = 2
        while x * x < n:
            if prime[x]:
                s = x * x   # Starting point at the sieve
                while s < n:
                    prime[s] = False
                    s += x
            x += 1

        return prime.count(True)

    '''
    [Leetcode: Medium] (46) Permutations, 2/20/2016
    Given a collection of distinct numbers, return all possible permutations.

    For example,
    [1,2,3] have the following permutations:
    [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], and [3,2,1]
    '''
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) < 1:
            return [nums]

        '''
        # Another solution
        def permuteHelper(nums, p, permutations):
            if not nums:
                permutations.append(p)
            else:
                for n in nums:
                    permuteHelper(nums-set([n]), p+[n], permutations)

        permutations = []
        permuteHelper(set(nums), [], permutations)
        return permutations
        '''

        res = []
        for i in range(len(nums)):
            for j in self.permute(nums[:i] + nums[i+1:]):
                print [nums[i]], j, [nums[i]] + j
                res.append([nums[i]] + j)
        return res

    '''
    [Leetcode: Medium] (319) Bulb Switcher, 2/20/2016
    There are n bulbs that are initially off. You first turn on all the bulbs.
    Then, you turn off every second bulb. On the third round,
    you toggle every third bulb (turning on if it's off or turning off if it's on).
    For the nth round, you only toggle the last bulb. Find how many bulbs are on after n rounds.

    For example, given n = 3.
    At first, the three bulbs are [off, off, off].
    After first round, the three bulbs are [on, on, on].
    After second round, the three bulbs are [on, off, on].
    After third round, the three bulbs are [on, off, off].
    So you should return 1, because there is only one bulb is on.
    '''

    def bulbSwitch(self, n):
        """
        :type n: int
        :rtype: int
        """
        # on: True, off: False

        '''
        # Naive approach:
        #   1) Turn on and off the bulbs as instructions
        #   2) Count the bulb-ons

        if n == 0:
            return 0

        bulbs = [False] * n

        # i: round, j: bulb location
        for i in range(1, n+1):
            j = i
            while j <= n:
                if bulbs[j-1] == False:
                    bulbs[j-1] = True
                else:
                    bulbs[j-1] = False
                j += i

        return bulbs.count(True)
        '''

        # Let's focus on the number of toggles for each bulb at each (R)ound
        #   1) 1st (R): all bulbs would be toggled (i.e., all turned on)
        #   2) 2nd (R): bulbs of which multiple of 2 would be toggled.
        #   3) 3rd (R): bulbs of which multiple of 3 would be toggled.
        #   ...
        #   n) nth (R): bulbs of which multiple of n would be toggled.

        # Observation
        #   1) The (j)th bulb would be toggled (touched) iff (i)th round is (j)'s divisor
        #      ex) 6th bulb can be toggled at 1st, 2nd, 3rd, and 6th round
        #   2) In other words, the number of toggles represents the number of divisors
        #   3) The number of divisors can be computed with prime factorization
        #      Generally, n = p_1^n_1 * p_2^n_2 * ... * p_k^n_k then pi(n) = (n_1+1)*...*(n_k+1)
        #      ex) 12 = 2^2 * 3^1; hence 12 has (2+1)*(1+1) = 6 divisors (1,2,3,4,6,12)
        #   4) Observe that even number of toggles leads to turn off the bulb in the end,
        #      Because, the bulb is turned on in the beginning. (on->off->on->off->...)
        #      We are looking for the number of bulbs to be turn on at the final round
        #   5) The only case that the number of divisors can be odd is that j is a square number
        #      ex) 1=1^2, 4=2^2, 9=3^2, 16=4^2, ... these bulbs would be touched 3 times (odd)
        #   6) Note that n^2k form will be interpreted as (n^(k*2)) eventually
        #      ex) 4^4 (256) has odd number of divisors, but it can be counted as 16^2

        # Conclusion
        #   -> Thus finding the number of toggles is identical with the problem to
        #      count square numbers, less than n

        '''
        # Method A
        toggles = 0
        i = 1

        while i*i <= n:
            toggles += 1
            i += 1

        return toggles
        '''

        # Method B
        import math
        return int(math.sqrt(n))

    '''
    [Leetcode: Easy] (160) Intersection of Two Linked Lists, 2/20/2016
    Write a program to find the node at which the intersection of two singly linked lists begins.
    For example, the following two linked lists:

    A:          a1 -> a2
                       -\
                         c1 -> c2 -> c3
                       -/
    B:     b1 -> b2 -> b3
    '''
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """

        # Either header is None, then no intersection found
        if headA == None or headB == None:
            return None

        ptrA, ptrB = headA, headB
        lenA, lenB = 0, 0

        # Walk two linked list until the end
        while ptrA is not None:
            lenA += 1
            ptrA = ptrA.next

        while ptrB is not None:
            lenB += 1
            ptrB = ptrB.next

        # Adjust the pointer to compare LL-A with LL-B
        diff = abs(lenA - lenB)
        ptrA, ptrB = headA, headB

        if lenA > lenB:
            for i in range(diff):
                ptrA = ptrA.next

        if lenA < lenB:
            for i in range(diff):
                ptrB = ptrB.next

        # Traverse each entry to find the intersection
        while ptrA != ptrB:
            ptrA = ptrA.next
            ptrB = ptrB.next

        return ptrA

    '''
    [Leetcode: Easy] (203) Remove Linked List Elements, 2/20/2016
    Remove all elements from a linked list of integers that have value val.

    Example
    Given: 1 --> 2 --> 6 --> 3 --> 4 --> 5 --> 6, val = 6
    Return: 1 --> 2 --> 3 --> 4 --> 5
    '''
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """

        # Case of empty linked list
        if head is None:
            return None

        # Initialize the pointers
        cur = head
        prev = cur

        while cur is not None:
            # 1. When the node to remove is head; adjust the head
            if cur == head and cur.val == val:
                head = cur.next
                cur = head
                prev = cur

            # 2. When the node to remove is tail; update the tail
            elif cur.next == None and cur.val == val:
                prev.next = None
                break

            # 3. General case to find the value to remove
            elif cur != head and cur.val == val:
                prev.next = cur.next
                cur = cur.next

            # 4. Otherwise go the the next node
            else:
                prev = cur
                cur = cur.next

        return head

    '''
    [Leetcode: Medium] (141) Linked List Cycle, 2/20/2016
    Given a linked list, determine if it has a cycle in it.

    Follow up:
    Can you solve it without using extra space?
    '''
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """

        # When the linked list is empty or has only a sinlge element
        if head is None or head.next is None:
            return False

        # Define two pointers that has different speed
        slow = head
        fast = head

        # If there is any cycle, at some point,
        # slow and fast pointer will meet each other
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True

        return False

    '''
    [Leetcode: Medium] (142) Linked List Cycle, 2/22/2016
    Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
    Note: Do not modify the linked list.

    Follow up:
    Can you solve it without using extra space?
    '''

    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # Define two pointers - (s)low and (f)ast (2x faster)
        # Let x: the distance to the node that starts a cycle
        # Let y: the distance from x to the point
        #        where two pointers meet for the first time in a cycle
        # Let t: the distance from y to t
        # Let s and f: the distance that slow/fast pointer moves respectively

        # Observations
        # (a) x + y = s
        # (b) x + y + k(t + y) = f = 2s (where k is # of cycles that f moves)
        # (c) (b)-(a) = k(t + y) = s = x + y
        # (d) From (b), Should be k = 1 and t = x

        # Solution
        # (a) Find the location (x + y) that two pointers meet for the first time
        # (b) Initialize the slow pointer to head again
        # (c) Start moving the s, and start moving the f from the distance (x + y)
        # (d) The node to begin a cycle would be the location when s and f meet due to t = x

        # Handling special case - the # of node is either 0 or 1
        if head is None or head.next is None:
            return None

        # Handling special case - head has the cycle (head == tail)
        if head == head.next or head == head.next.next:
            return head

        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break

        # If fast pointer reaches None, there is no detected cycle.
        # Otherwise, f is pointing to the location of (a)
        if slow == fast:
            # Initialize s pointer again as described in (b)
            slow = head

            # Find the location to begin a cycle (t=x) like (c) and (d)
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow

        return None

    '''
    [Leetcode: Medium] (92) Reverse Linked List II, 2/27/2016
    Reverse a linked list from position m to n. Do it in-place and in one-pass.

    For example:
    Given 1->2->3->4->5->NULL, m = 2 and n = 4,
    return 1->4->3->2->5->NULL.
    '''
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """

        # Handling weird cases
        if m > n or m <= 0:
            return None

        # If the number of node is less than 2 then return head
        if head == None or head.next == None or m == n:
            return head

        # Initialization
        prev = head
        cur = prev
        cnt = 1

        while cnt <= n:
            # Move forward two pointers until m
            if cnt <= m:
                # Store the pointers to be updated later on (at n)
                if cnt == m:
                    left_m = prev
                    right_n = cur
                prev = cur
                cur = cur.next

            # Reverse the pointers in the middle
            elif m < cnt < n:
                tmp = prev
                prev = cur
                cur = cur.next
                prev.next = tmp

            # Update two pointers left_m and right_n when arriving at the position n
            elif cnt == n:
                left_m.next = cur
                right_n.next = cur.next
                tmp = prev
                prev = cur
                cur = cur.next
                prev.next = tmp

            cnt += 1

        # Return head when m > 2, or prev (pointing to the new head)
        return head if m != 1 else prev

    '''
    [Leetcode: Easy] (303) Range Sum Query - Immutable, 2/27/2016
    Given nums = [-2, 0, 3, -5, 2, -1]
    sumRange(0, 2) -> 1
    sumRange(2, 5) -> -1
    sumRange(0, 5) -> -3
    Note
        You may assume that the array does not change.
        There are many calls to sumRange function.
    '''
    class NumArray(object):
        def __init__(self, nums):
            """
            initialize your data structure here.
            :type nums: List[int]
            """
            # Define the table for dynamic programming
            #    that has all sums from the first element to the current
            self.dp = [sum(nums[0:i+1]) for i in range(len(nums))]

        def sumRange(self, i, j):
            """
            sum of elements nums[i..j], inclusive.
            :type i: int
            :type j: int
            :rtype: int
            """
            # Sum of the range can be obtained by subtracting dp[j] - dp[i-1]
            return self.dp[j] if i == 0 else self.dp[j] - self.dp[i-1]

    '''
    [Leetcode: Easy] (198) House Robber, 3/3/2016

    You are a professional robber planning to rob houses along a street.
    Each house has a certain amount of money stashed, the only constraint stopping you from robbing
    each of them is that adjacent houses have security system connected and
    it will automatically contact the police if two adjacent houses were broken into on the same night.
    Given a list of non-negative integers representing the amount of money of each house,
    determine the maximum amount of money you can rob tonight without alerting the police.
    '''
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if len(nums) == 0:
            return 0

        if len(nums) == 1 or len(nums) == 2:
            return max(nums)

        # Initialize DP table
        dp = [nums[0], max(nums[0], nums[1])]

        # For each pair, check if which is bigger: dp[i-1], nums[i] + dp[j]
        for i in range(2, len(nums)):
            dp.append(-1)
            for j in range(i-1):
                dp[i] = max(dp[i-1], nums[i] + dp[j])

        return max(dp[-1], dp[-2])

    '''
    [Leetcode: Medium] (136) Single Number I, 3/4/2016

    Given an array of integers, every element appears twice except for one. Find that single one.
    Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
    '''
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        # By looking at the numbers in nums one by one
        # If we have not seen the number, push it to the stack (res)
        # Otherwise pop it. Only single element remains there in the end!

        '''
        # Method A using a list
        res = list()
        for i in range(len(nums)):
            if not nums[i] in res:
                res.append(nums[i])
            else:
                res.remove(nums[i])

        return res[0]
        '''

        # Method B using a dictionary (faster due to hash)
        ht = dict()

        for i in range(len(nums)):
            if not nums[i] in ht:
                ht[nums[i]] = 1
            else:
                ht.pop(nums[i])

        return ht.keys()[0]

    '''
    [Leetcode: Medium] (137) Single Number II, 3/4/2016

    Given an array of integers, every element appears three times except for one. Find that single one.
    Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
    '''
    def singleNumber2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ht = dict()

        # Very similar to 136, but we don't need to care # of times
        # If the number is more than 1, just pop off of the hash table!
        for i in range(len(nums)):
            if nums[i] not in ht:
                ht[nums[i]] = 1
            elif ht[nums[i]] == 2:
                ht.pop(nums[i])
            else:
                ht[nums[i]] += 1

        return ht.keys()[0]

    '''
    [Leetcode: Medium] (260) Single Number III, 3/4/2016

    Given an array of numbers nums, in which exactly two elements appear only once and
    all the other elements appear exactly twice. Find the two elements that appear only once.
    For example:
    Given nums = [1, 2, 1, 3, 2, 5], return [3, 5].

    Note:
    The order of the result is not important. So in the above example, [5, 3] is also correct.
    Your algorithm should run in linear runtime complexity. Could you implement it using only constant space complexity?
    '''
    def singleNumber3(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        # This problem can be easily solved using exactly the same approach with 136 and 137!
        ht = dict()

        for i in range(len(nums)):
            if nums[i] not in ht:
                ht[nums[i]] = 1
            else:
                ht.pop(nums[i])

        return ht.keys()

    '''
    [Leetcode: Medium] (125) Valid Palindrome, 3/4/2016

    Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
    For example,
        "A man, a plan, a canal: Panama" is a palindrome.
        "race a car" is not a palindrome.
    '''
    def isPalindrome2(self, s):
        """
        :type s: str
        :rtype: bool
        """

        '''
        # The following is a valid answer, but somehow leetcode complains the time constraint.

        def isPalindromeHelper(s):
            res = True
            for i in range(len(s)//2):
                if s[i] != s[-i-1]:
                    res = False
            return res

        def sanitizer(s):
            res = ''
            for i in range(len(s)):
                if 'a' <= s[i] < 'z' or 'A' <= s[i] <= 'Z' or '0' <= s[i] <= '9':
                    res += s[i]
            return res.lower()

        if len(sanitizer(s)) == 0:
            return True

        return isPalindromeHelper(sanitizer(s))
        '''

        # Define two index pointers (start, end)
        si, ei = 0, len(s)-1

        while si < ei:
            # Move to a valid starting/ending index until consecutive comparison can be made
            while si < ei and not s[si].isalnum():
                si += 1
            while si < ei and not s[ei].isalnum():
                ei -= 1

            # Compare two characters where two pointers point to
            if si < ei and s[si].lower() != s[ei].lower():
                return False

            si += 1
            ei -= 1

        return True

    '''
    [Leetcode: Easy] (88) Merged Sorted Array, 3/5/2016

    Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
    You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements
    from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.
    '''
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """

        # Define index i,j,k for nums1, nums2, and nums1+nums2
        i, j = m-1, n-1
        k = m+n-1

        # Switch the positions
        while i>=0 and j>=0:
            if nums2[j] > nums1[i]:
                nums1[k] = nums2[j]
                j -= 1
            else:   # nums2[j] < nums1[i]
                nums1[k] = nums1[i]
                i -= 1
            k -= 1

        # Fill out the remaining elements in nums1
        for x in range(j+1):
            nums1[x] = nums2[x]

    '''
    [Leetcode: Easy] (118) Pascal's Triangle, 3/13/2016

    Given numRows, generate the first numRows of Pascal's triangle.
    For example, given numRows = 5, return
    [[1], [1,1], [1,2,1], [1,3,3,1], [1,4,6,4,1]]
    '''
    def generatePascal(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """

        def generateRow(row, curList):
            if row < 3:
                return [1] * row

            # Generate the Pascal's triangle at the row
            #  a) Get the last element of the list
            #  b) Compute all elements that consist of the sum of two consecutive numbers
            #  c) Attach the first and the last element which are always 1
            return [1] + [curList[-1][x]+curList[-1][x+1] for x in range(len(curList[-1])-1)] + [1]


        if numRows <= 0:
            return []

        ans = []

        for i in range(1, numRows + 1):
            ans.append(generateRow(i, ans))

        return ans

    '''
    [Leetcode: Easy] (67) Add Binary, 3/13/2016

    Given two binary strings, return their sum (also a binary string).
    For example, a = "11", b = "1", Return "100".
    '''
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """

        def digitMatch(s, n):
            return s.zfill(n)

        if a == '0' and b == '0':
            return '0'

        # Sanitizing the size of two digits
        if len(a) > len(b):
            b = digitMatch(b, len(a))
        else:
            a = digitMatch(a, len(b))

        ans = ''
        carry = 0

        for i in range(len(a)-1, -1, -1):
            # Check out output digit and carry together
            digitsum = int(a[i]) + int(b[i]) + carry
            ans += str(digitsum % 2)

            if digitsum >= 2:
                carry = 1
            else:
                carry = 0

        # If there is a carry left in the end, append it
        if carry == 1:
            ans += '1'

        # Return the answer in reverse order
        return ans[::-1]

    '''
    [Leetcode: Easy] (26) Remove Duplicates from Sorted Array, 3/13/2016

    Given a sorted array, remove the duplicates in place
    such that each element appear only once and return the new length.
    Do not allocate extra space for another array, you must do this in place with constant memory.

    For example,
    Given input array nums = [1,1,2],
    Your function should return length = 2, with the first two elements of nums
    being 1 and 2 respectively. It doesn't matter what you leave beyond the new length.
    '''
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if len(nums) == 0:
            return 0

        # Initialize the pointer to store a unique element
        pos = 1

        # Keep checking if nums[i-1] == nums[i]
        for cur in range(1, len(nums)):

            # If the same, move the cur ptr
            if nums[cur-1] == nums[cur]:
                pass

            # If not, store it to the position of pos ptr
            else:
                nums[pos] = nums[cur]
                pos += 1

        return pos

    '''
    [Leetcode: Easy] (20) Valid Parentheses, 3/13/2016

    Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
    determine if the input string is valid.

    The brackets must close in the correct order,
    "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
    '''
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """

        if len(s) == 0:
            return True

        pair = {')':'(', '}':'{', ']':'['}
        stack = []

        for p in s:

            # Keep pushing an open parenthesis on the stack
            if p == '(' or p == '{' or p == '[':
                stack.append(p)

            else:

                # If something comes in without open parenthesis, invalid
                if len(stack) == 0:
                    return False

                # If close parenthesis matches another pair, pop it
                if pair[p] == stack[-1]:
                    stack.pop()

                # If any of them does not match, invalid
                else:
                    return False

        return len(stack) == 0

    '''
    [Leetcode: Medium] (338) Counting Bits, 3/19/2016

    Given a non negative integer number num. For every numbers i in the range 0 <= i <= num
    calculate the number of 1's in their binary representation and return them as an array.
    Example:
    For num = 5 you should return [0,1,1,2,1,2].
    '''
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """

        '''
        # With built-in func in python
        res = []
        for i in range(num + 1):
            res.append(bin(i).count('1'))

        return res
        '''

        # Compute the biggest power less than n
        power = 0
        tmp = num

        while tmp > 0:
            tmp //= 2
            power += 1

        # Observe that the first 2^(n-1) consist of the next 2^(n-1) elements of 2^n
        # i.e. [0,1,1,2] + [1,2,2,3] = [0,1,1,2,1,2,2,3]
        res = [0]
        if num > 0:
            for i in range(power):
                for j in range(len(res)):
                    res.append(res[j] + 1)

        return res[:num+1]

    '''
    [Leetcode: Hard] (146) LRU Cache, 3/19/2016

    Design and implement a data structure for Least Recently Used (LRU) cache.
    It should support the following operations: get and set.

    get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
    set(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity,
    it should invalidate the least recently used item before inserting a new item.
    '''
    class LRUCache(object):

        def __init__(self, capacity):
            """
            :type capacity: int
            """
            self.capacity = capacity
            self.cache = dict()
            self.queue = []

        def _isKey(self, key):
            return True if key in self.cache.keys() else False

        def get(self, key):
            """
            :rtype: int
            """
            if len(self.cache) == 0:
                return -1

            if self._isKey(key):
                self.queue.remove(key)
                self.queue.append(key)
                return self.cache[key]
            else:
                return -1


        def set(self, key, value):
            """
            :type key: int
            :type value: int
            :rtype: nothing
            """
            if not self._isKey(key):
                if len(self.cache) == self.capacity:
                    self.cache.pop(self.queue[0])
                    self.queue = self.queue[1:]
            else:
                self.queue.remove(key)

            self.cache[key] = value
            self.queue.append(key)

    '''
    [Leetcode: Easy] (172) Factorial Trailing Zeroes, 3/14/2016

    Given an integer n, return the number of trailing zeroes in n!.
    Note: Your solution should be in logarithmic time complexity.
    '''
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """

        # 2 * 5 = 10; increases a single digit simply

        # a) Every multiple of 5 increases a digit
        # b) x = (even) * (5 ^ i) where x < n, i=2,3,4, ...
        # Ref. http://www.purplemath.com/modules/factzero.htm

        trails = 0

        i = 1
        while (5 ** i) <= n:
            trails += (n // (5 ** i))
            i += 1

        return trails

    '''
    [Leetcode: Easy] (38) Count and Say, 3/14/2016

    The count-and-say sequence is the sequence of integers beginning as follows:
    1, 11, 21, 1211, 111221, ...

    1 is read off as "one 1" or 11.
    11 is read off as "two 1s" or 21.
    21 is read off as "one 2, then one 1" or 1211
    '''
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """

        def readNext(s):
            new_cnt = ''
            cnt = 1

            # Read off the next number
            for i in range(1, len(s)):
                if s[i-1] == s[i]:
                    cnt += 1
                else:
                    new_cnt += str(cnt) + s[i-1]
                    cnt = 1

            # Read the last (remaining) element
            new_cnt += str(cnt) + s[i]
            return new_cnt

        # Initialize the first element
        cntArray = ['1', '11']

        # Generate 'count-and-say' numbers
        for i in range(2, n):
            cntArray.append(readNext(cntArray[-1]))

        return cntArray[n-1]

    '''
    [Leetcode: Easy] (112) Path Sum, 3/15/2016

    Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

    For example:
    Given the below binary tree and sum = 22,
                  5
                 / \
                4   8
               /   / \
              11  13  4
             /  \      \
            7    2      1
    return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
    '''
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """

        def curSum(root, sum, s):

            # Base case; reaches the bottom w/o obtaining a given sum
            if root == None:
                return False

            # Return true iff the node is a leaf and the condition is met
            elif root.left == None and root.right == None and root.val + s == sum:
                return True

            # Recursively search the tree
            else:
                return curSum(root.left, sum, root.val + s) or curSum(root.right, sum, root.val + s)

        return curSum(root, sum, 0)

    '''
    [Leetcode: Easy] (19) Remove Nth Node From End of List, 3/16/2016

    Given a linked list, remove the nth node from the end of list and return its head.

    For example,
    Given linked list: 1->2->3->4->5, and n = 2.
    After removing the second node from the end, the linked list becomes 1->2->3->5.
    Note:
        Given n will always be valid.
        Try to do this in one pass.
    '''
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        if head == None or n < 1:
            return None

        ptr = head
        total = 0

        # Obtain the total number of nodes in a linked list
        while ptr:
            total += 1
            ptr = ptr.next

        # Special cases when removing the first or the last node
        if total == 1:
            return None

        if total == n:
            return head.next

        # Move the pointer to the pre-position of the node to be removed
        ptr = head
        for i in range(total-(n+1)):
            ptr = ptr.next

        # If removing node is the last one
        if n == 1:
            ptr.next = None

        # Update the pointer otherwise
        else:
            newNext = ptr.next.next
            ptr.next = newNext

        return head

    '''
    [Leetcode: Easy] (58) Length of Last Word, 3/16/2016

    Given a string s consists of upper/lower-case alphabets and
    empty space characters ' ', return the length of last word in the string.
    If the last word does not exist, return 0.
    Note: A word is defined as a character sequence consists of
    non-space characters only.

    For example,
    Given s = "Hello World",
    return 5.
    '''
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """

        if len(s) == 0:
            return 0

        # Split all words with the delimeter as space
        all_words = [x for x in s.split(' ') if len(x) > 0]

        # Return the length of the last word, or 0 if empty
        return len(all_words[-1]) if len(all_words) > 0 else 0

    '''
    [Leetcode: Easy] (223) Rectangle Area, 3/19/2016

    Find the total area covered by two rectilinear rectangles in a 2D plane.
    Each rectangle is defined by its bottom left corner and top right corner.
    Assume that the total area is never beyond the maximum possible value of int
    '''
    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """

        area = (C-A)*(D-B)+(G-E)*(H-F)

        # Case 1) two squares has no region to share
        if (G-E)+(C-A) <= abs(max(C,G)-min(A,E)) or (D-B) + (H-F) <= abs(max(D,H)-min(B,F)):
            return area

        # Case 2) one square includes another
        elif (E<A and C<G and F<B and D<H) or (E>A and C>G and F>B and D>H):
            return (max(C,G)-min(A,E))*(max(D,H)-min(B,F))

        # Case 3) two squares has some region to share otherwise
        else:
            intersect = min(G-A,C-E,C-A,G-E) * min(D-F,H-B,D-B,H-F)
            return area - intersect

    '''
    [Leetcode: Medium] (89) Gray Code, 3/26/2016

    The gray code is a binary numeral system where two successive values differ in only one bit.
    Given a non-negative integer n representing the total number of bits in the code, print the sequence of gray code. A gray code sequence must begin with 0.
    For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:
    00 - 0
    01 - 1
    11 - 3
    10 - 2
    '''
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """

        if n == 0:
            return [0]

        seq = ['0', '1']

        # Key element is to observe that the next series should flip previous set and proceed.
        # [0,1] -> 1,0 -> 11,10 -> [3,2]
        # [0,1,3,2] -> 00,01,11,10 -> 10,11,01,00 -> 110,111,101,100 -> [6,7,5,4]
        # [0,1,3,2,6,7,5,4] -> 1100,1101,1111,1110,1010,1011,1001,1000 -> [12,13,15,14,10,11,9,8]
        for i in range(1, n):
            seq += [('1'+x) for x in seq[::-1]]
            seq = [x.zfill(i+1) for x in seq]

        return [int('0b' + str(int(x)), 2) for x in seq]

    '''
    [Leetcode: Easy] (299) Bulls and Cows, 3/26/2016

    You are playing the following Bulls and Cows game with your friend: You write down a number
    and ask your friend to guess what the number is. Each time your friend makes a guess,
    you provide a hint that indicates how many digits in said guess match your secret number exactly
    in both digit and position (called "bulls") and how many digits match the secret number
    but locate in the wrong position (called "cows"). Your friend will use successive guesses
    and hints to eventually derive the secret number.

    For example:
    Secret number:  "1807"
    Friend's guess: "7810"

    Hint: 1 bull and 3 cows. (The bull is 8, the cows are 0, 1 and 7.)
    Write a function to return a hint according to the secret number and friend's guess,
    use A to indicate the bulls and B to indicate the cows. In the above example, your function should return "1A3B".

    Please note that both secret number and friend's guess may contain duplicate digits, for example:

    Secret number:  "1123"
    Friend's guess: "0111"
    In this case, the 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow, and your function should return "1A1B".
    You may assume that the secret number and your friend's guess only contain digits,
    and their lengths are always equal.
    '''

    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """

        if len(secret) == 0 or len(secret) != len(guess):
            return ''

        '''
        # Naive approach O(N^2) - Time Limit
        bulls, cows = 0, 0

        secrets = [s for s in secret]
        guesses = [g for g in guess]

        for i in range(len(guess)):
            if guess[i] == secrets[i]:
                bulls += 1
                secrets[i] = '*'
                guesses[i] = '*'

        for j in range(len(guess)):
            if guesses[j] == '*':
                pass
            else:
                if guesses[j] in secrets:
                    cows += 1
                    secrets[secrets.index(guesses[j])] = '*'
                    guesses[j] = '*'

        return str(bulls) + 'A' + str(cows) + 'B'
        '''

        s_bag = {}
        g_bag = {}
        bulls = 0

        # O(N) solution
        for i in range(len(guess)):

            # Checks all positions and count bulls
            if secret[i] == guess[i]:
                bulls += 1

            # Otherwise count other numbers that does not match in each bag
            else:
                if secret[i] not in s_bag:
                    s_bag[secret[i]] = 1
                else:
                    s_bag[secret[i]] += 1
                if guess[i] not in g_bag:
                    g_bag[guess[i]] = 1
                else:
                    g_bag[guess[i]] += 1

        # Count cows: the min of which both g_bag and s_bag contains
        cows = sum([min(s_bag[x], g_bag[x]) for x in g_bag if x in s_bag])
        return str(bulls) + 'A' + str(cows) + 'B'

    '''
    [Leetcode: Easy] (257) Binary Tree Paths, 4/2/2016

    Given a binary tree, return all root-to-leaf paths.
    For example, given the following binary tree:

       1
     /   \
    2     3
     \
      5
    All root-to-leaf paths are:

    ["1->2->5", "1->3"]
    '''
    def binaryTreePaths(self, root):

        def dfsHelper(root, path, paths):
            # Case 1: Node is a leaf
            if root.left == None and root.right == None:
                paths.append(path)

            # Case 2: Node has a left child
            if root.left:
                dfsHelper(root.left, path + "->" + str(root.left.val), paths)

            # Case 3: Node has a right child
            if root.right:
                dfsHelper(root.right, path + "->" + str(root.right.val), paths)

        if root is None:
            return []

        paths = []
        dfsHelper(root, str(root.val), paths)
        return paths

    '''
    [Leetcode: Medium] (322) Coin Change, 4/2/2016

    You are given coins of different denominations and a total amount of money amount.
    Write a function to compute the fewest number of coins that you need to make up that amount.
    If that amount of money cannot be made up by any combination of the coins, return -1.

    Example 1:
    coins = [1, 2, 5], amount = 11
    return 3 (11 = 5 + 5 + 1)

    Example 2:
    coins = [2], amount = 3
    return -1.

    Note:
    You may assume that you have an infinite number of each kind of coin.
    '''
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """

        if amount == 0:
            return 0

        if min(coins) > amount:
            return -1

        # Initialize DP table
        dp = [amount+1] * amount

        # Construct DP table
        # dp[next] = min(dp[current coin - (all coins)]) + 1
        for idx in range(amount):
            coin = idx+1
            if coin in coins:
                dp[idx] = 1
            else:
                chk = [coin-x for x in coins if coin-x>=0]
                if len(chk) != 0:
                    dp[idx] = min([dp[x-1] for x in chk]) + 1

        # return dp[amount-1] if the value is available
        return -1 if dp[amount-1] > amount else dp[amount-1]

    '''
    [Leetcode: Easy] (290) Word Pattern, 4/4/2016
    Given a pattern and a string str, find if str follows the same pattern.

    Here follow means a full match, such that there is a bijection
    between a letter in pattern and a non-empty word in str.

    Examples:
    pattern = "abba", str = "dog cat cat dog" should return true.
    pattern = "abba", str = "dog cat cat fish" should return false.
    pattern = "aaaa", str = "dog cat cat dog" should return false.
    pattern = "abba", str = "dog dog dog dog" should return false.
    Notes:
    You may assume pattern contains only lowercase letters,
    and str contains lowercase letters separated by a single space.
    '''
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """

        p = list(pattern)
        strp = str.split(' ')

        print p
        print strp

        if len(p) != len(strp):
            return False

        # Define two check bags
        chkbag1 = {}
        chkbag2 = {}

        ans = True

        # Check if a pattern needs to map only a single word
        # Note that a sinlge word also maps a single pattern

        for idx in range(len(p)):

            if p[idx] not in chkbag1:
                chkbag1[p[idx]] = strp[idx]

            if strp[idx] not in chkbag2:
                chkbag2[strp[idx]] = p[idx]

            if p[idx] in chkbag1:
                if chkbag1[p[idx]] != strp[idx]:
                    ans = False

            if strp[idx] in chkbag2:
                if chkbag2[strp[idx]] != p[idx]:
                    ans = False

        return ans

    '''
    [Leetcode: Hard] (87) Scramble String, 4/16/2016
    Given a string s1, we may represent it as a binary tree by partitioning it to two non-empty substrings recursively.

    Below is one possible representation of s1 = "great":

        great
       /    \
      gr    eat
     / \    /  \
    g   r  e   at
               / \
              a   t
    To scramble the string, we may choose any non-leaf node and swap its two children.

    For example, if we choose the node "gr" and swap its two children, it produces a scrambled string "rgeat".

        rgeat
       /    \
      rg    eat
     / \    /  \
    r   g  e   at
               / \
              a   t
    We say that "rgeat" is a scrambled string of "great".

    Similarly, if we continue to swap the children of nodes "eat" and "at", it produces a scrambled string "rgtae".

        rgtae
       /    \
      rg    tae
     / \    /  \
    r   g  ta  e
           / \
          t   a
    We say that "rgtae" is a scrambled string of "great".

    Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1
    '''
    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """

        # See if the length of two strings are same
        if len(s1) != len(s2):
            return False

        # See if two strings are same
        if s1 == s2:
            return True

        # See if the number of alphabet in both strings are same
        for i in 'abcdefghijklmnopqrstuvwxyz':
            if s1.count(i) != s2.count(i):
                return False

        str_len = len(s1)

        # Recursively check if the subsets of s1 can produce the scramble string s2
        for i in range(1, str_len):

            # Forward check
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
                return True

            # Backward check
            if self.isScramble(s1[:i], s2[-i:]) and self.isScramble(s1[i:], s2[:str_len-i]):
                return True

        return False

    '''
    [Leetcode: Medium] (113) Path Sum II, 4/16/2016
    Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

    For example:
    Given the below binary tree and sum = 22,
                  5
                 / \
                4   8
               /   / \
              11  13  4
             /  \    / \
            7    2  5   1

    return [[5,4,11,2], [5,8,4,5]]
    '''
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """

        # Visit all root-to-leaf paths to check if the path has the sum recursively
        def pathSumHelper(root, s, cur_path, res):

            # Base case: the current sum satisfies the condition, append the path
            if root.left == None and root.right == None:
                if s == sum:
                    res.append(cur_path)

            # If there is a either left or right node, visit to the node
            if root.left:
                pathSumHelper(root.left, s + root.left.val, cur_path + [root.left.val], res)

            if root.right:
                pathSumHelper(root.right, s + root.right.val, cur_path + [root.right.val], res)

        if root == None:
            return []

        res = []
        pathSumHelper(root, root.val, [root.val], res)

        return res

    '''
    [Leetcode: Easy] (344) Reverse String, 4/23/2016
    Write a function that takes a string as input and returns the string reversed.

    Example:
    Given s = "hello", return "olleh".
    '''
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """

        str_list = list(s)

        # Simplest solution
        # return s[::-1]

        # Change the positions in order until the middle point
        for i in range(len(s)//2):
            tmp = str_list[i]
            str_list[i] = str_list[-i-1]
            str_list[-i-1] = tmp

        return "".join(str_list)

    '''
    [Leetcode: Medium] (108) Convert Sorted Array to Binary Search Tree, 4/23/2016
    Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
    '''
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        def buildTree(nums):

            # Base Case
            if not nums:
                return None

            # Every call, build left and right side of the tree
            mid = len(nums)//2
            root = basicds.TreeNode(nums[mid])
            root.left = buildTree(nums[:mid])
            root.right = buildTree(nums[mid+1:])

            return root

        return buildTree(nums)

    '''
    [Leetcode: Medium] (116) Populating Next Right Pointers in Each Node, 4/23/2016
    Given a binary tree

        struct TreeLinkNode {
          TreeLinkNode *left;
          TreeLinkNode *right;
          TreeLinkNode *next;
        }
    Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

    Initially, all next pointers are set to NULL.

    Note:

    You may only use constant extra space.
    You may assume that it is a perfect binary tree (ie, all leaves are at the same level, and every parent has two children).
    For example,
    Given the following perfect binary tree,
             1
           /  \
          2    3
         / \  / \
        4  5  6  7
    After calling your function, the tree should look like:
             1 -> NULL
           /  \
          2 -> 3 -> NULL
         / \  / \
        4->5->6->7 -> NULL
    '''
    def connect(self, root):
        """
        :type root: TreeLinkNode
        :rtype: nothing
        """

        if root == None or root.right == None:
            return None

        # Observation
        # (a) the root-left-next always connects to root-right
        # (b) if root has next, root-right-next connects to root-next-left
        root.left.next = root.right
        if root.next:
           root.right.next = root.next.left

        # Iteratively connect all nodes at the same layer
        self.connect(root.left)
        self.connect(root.right)

    '''
    [Leetcode: Hard] (117) Populating Next Right Pointers in Each Node II, 4/23/2016
    Follow up for problem 116 (Generalization)
    What if the given tree could be any binary tree? Would your previous solution still work?

    Note: you may only use constant extra space.

    For example,
    Given the following binary tree,
             1
           /  \
          2    3
         / \    \
        4   5    7
    After calling your function, the tree should look like:
             1 -> NULL
           /  \
          2 -> 3 -> NULL
         / \    \
        4-> 5 -> 7 -> NULL
    '''
    def connect2(self, root):
        """
        :type root: TreeLinkNode
        :rtype: nothing
        """
        # Check if the next node is available in the same layer
        def get_next(node):
            while True:
                if node.left:
                    return node.left
                elif node.right:
                    return node.right
                else:
                    if node.next:
                        node = node.next
                    else:
                        return None

        # Base case
        if root == None:
            return None

        # When there is only a left node
        if root.left and not root.right:
            if root.next:
                root.left.next = get_next(root.next)

        # When there is a right node
        if root.right:
            if root.next:
                root.right.next = get_next(root.next)

        # Both children are available
        if root.left and root.right:
            root.left.next = root.right

        # IDEA: Construct the link from the right!!
        self.connect2(root.right)
        self.connect2(root.left)

    '''
    [Leetcode: Easy] (342) Power of Four, 4/25/2016
    Given an integer (signed 32 bits), write a function to check whether it is a power of 4.

    Example:
    Given num = 16, return true. Given num = 5, return false.

    Follow up: Could you solve it without loops/recursion?
    '''
    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """

        # Power can't be 0 or negative
        if num <= 0:
            return False

        '''
        Observation
        1) Only the first digit should be 1, followed by zeroes
            4^0 = 1 = 2^0 (1)
            4^1 = 4  = 2^2 (100)
            4^2 = 16  = 2^4 (10000)
            4^3 = 64  = 2^6 (1000000)
            4^4 = 256 = 2^8 (100000000)
        2) The number of zeroes is always even
        '''
        b = bin(num).split('0b')[1]
        nz = b.count('0')

        return b[0] == '1' and nz == len(b)-1 and nz%2 == 0

    '''
    [Leetcode: Easy] (345) Power of Four, 4/27/2016
    Write a function that takes a string as input and reverse only the vowels of a string.

    Example 1:
    Given s = "hello", return "holle".

    Example 2:
    Given s = "leetcode", return "leotcede".
    '''
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """

        '''
        # Correct implementation but time limit excceeded..

        vowels = {}

        # Build the table for vowel positions
        for i in range(len(s)):
            if s[i] in 'aeiouAEIOU':
                vowels[i] = s[i]

        if len(vowels) == 0:
            return s

        strs = list(s)
        v_idxes = list(sorted(vowels))

        # Exchange the vowel positions
        for i in range(len(v_idxes)//2):
            tmp = vowels[v_idxes[i]]
            vowels[v_idxes[i]] = vowels[v_idxes[-i-1]]
            vowels[v_idxes[-i-1]] = tmp

        # Combine the final strings with the exchanged vowels
        for j in range(len(strs)):
            if j in v_idxes:
                strs[j] = vowels[j]

        return ''.join(strs)
        '''

        '''
        # Did it in a similar fashion; does not work again!

        idxes, vowels = [], []
        strs = list(s)
        ans = ''

        for idx, v in enumerate(s):
            if v in 'aeiouAEIOU':
                idxes.append(idx)
                vowels.append(v)

        vowels.reverse()

        for i in range(len(strs)):
            if i in idxes:
                ans += vowels[idxes.index(i)]
            else:
                ans += s[i]

        return ans
        '''

        # Set the two pointers from the front and the back
        fp, bp = 0, len(s)-1

        vows = 'aeiouAEIOU'
        strs = list(s)

        # Whenever vowels are found, exchange the two
        while fp < bp:
            if s[fp] in vows and s[bp] in vows:
                strs[fp], strs[bp] = strs[bp], strs[fp]
                fp += 1
                bp -= 1

            # Otherwise move the pointers
            elif s[fp] not in vows:
                strs[fp] = s[fp]
                fp += 1

            elif s[bp] not in vows:
                bp -= 1

        return ''.join(strs)

    '''
    [Leetcode: Easy] (171) Excel Sheet Column Number, 4/27/2016
    Related to question Excel Sheet Column Title
    Given a column title as appear in an Excel sheet, return its corresponding column number.

    For example:
        A -> 1
        B -> 2
        C -> 3
        ...
        Z -> 26
        AA -> 27
        AB -> 28
    '''
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """

        '''
        u_cases = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        case_val = {}

        for i in range(len(u_cases)):
            case_val[u_cases[i]] = i+1

        # 26 number system
        return sum([(26**x) * case_val[s[-x-1]] for x in range(len(s))])
        '''

        # Faster one
        return sum([(26**digit) * (ord(x)-64) for (digit, x) in enumerate(s[::-1])])

    '''
    [Leetcode: Meidum] (78) Subsets, 5/7/2016
    Given a set of distinct integers, nums, return all possible subsets.

    Note:
    Elements in a subset must be in non-descending order.
    The solution set must not contain duplicate subsets.

    For example,
    If nums = [1,2,3], a solution is:
    [[3],[1],[2],[1,2,3],[1,3],[2,3],[1,2],[]]
    '''
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        '''
        # Using itertools
        import itertools
        sol = []

        for i in range(len(nums)+1):
            comb = [x for x in itertools.combinations(nums, i)]

            for j in range(len(comb)):
                sol.append(sorted(list(comb[j])))

        return sol
        '''

        subsets = [[]]
        nums = sorted(nums)

        # Iteratively add subsets only if the subset is not in the current subsets
        for i in range(len(nums)):
            subsets += [subset+[nums[i]] for subset in subsets if subset+[nums[i]] not in subsets]

        return subsets

    '''
    [Leetcode: Meidum] (90) Subsets II, 5/7/2016
    Given a collection of integers that might contain duplicates, nums, return all possible subsets.

    Note:
    Elements in a subset must be in non-descending order.
    The solution set must not contain duplicate subsets.

    For example,
    If nums = [1,2,2], a solution is:
    [[2],[1],[1,2,2],[2,2],[1,2],[]]
    '''
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        nums = sorted(nums)
        subsets = [[]]

        # Similar to the problem 78
        for i in range(len(nums)):
            subsets += [subset+[nums[i]] for subset in subsets if subset+[nums[i]] not in subsets]

        return subsets

    '''
    [Leetcode: Hard] (149) Max Points on a Line, 5/7/2016
    Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.
    '''
    def maxPoints(self, points):
        def getSlope(pt1, pt2):
            # Parallel line to x axis
            if pt1.x - pt2.x == 0:
                slope = 'Inf'

            # General cases
            else:
                slope = (pt1.y - pt2.y) / float(pt1.x - pt2.x)

            return slope

        def isDupPoint(pt1, pt2):
            return pt1.x == pt2.x and pt1.y == pt2.y

        num_pts = len(points)
        if num_pts < 3:
            return num_pts

        max_pts = 0

        # If the slopes of the two points are same,
        # they lie on the same straight line.
        for i in range(num_pts):
            slopes = {}
            dup_pt = 0

            for j in range(i+1, num_pts):
                # If two points are identical, just count one
                if isDupPoint(points[i], points[j]):
                    dup_pt += 1
                    continue
                # Get the slope and maintain it to the hash table
                else:
                    s = getSlope(points[i], points[j])
                    if s not in slopes:
                        slopes[s] = 2
                    else:
                        slopes[s] += 1

            max_pts = max(max_pts, max(slopes.values())+dup_pt) if slopes else max(max_pts, dup_pt+1)

        return max_pts

    '''
    [Leetcode: Hard] (239) Sliding Window Maximum, 5/14/2016
    Given an array nums, there is a sliding window of size k which is moving from the very left of the array
    to the very right. You can only see the k numbers in the window.
    Each time the sliding window moves right by one position.

    For example,
    Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.

    Window position                Max
    ---------------               -----
    [1  3  -1] -3  5  3  6  7       3
     1 [3  -1  -3] 5  3  6  7       3
     1  3 [-1  -3  5] 3  6  7       5
     1  3  -1 [-3  5  3] 6  7       5
     1  3  -1  -3 [5  3  6] 7       6
     1  3  -1  -3  5 [3  6  7]      7
    Therefore, return the max sliding window as [3,3,5,5,6,7].

    Note:
    You may assume k is always valid, ie: 1 ≤ k ≤ input array's size for non-empty array.

    Follow up:
    Could you solve it in linear time?
    '''
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """

        ans = []

        if len(nums) == 0 or k == 0:
            return []

        # Get the max element from the range nums[i:i+k]
        for i in range(len(nums)-k+1):
            ans.append(max(nums[i:i+k]))

        return ans

    '''
    [Leetcode: Medium] (347) STop K Frequent Elements, 5/14/2016
    Given a non-empty array of integers, return the k most frequent elements.

    For example,
    Given [1,1,1,2,2,3] and k = 2, return [1,2].

    Note:
    You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
    Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
    '''
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """

        uniq_nums = list(set(nums))
        num_cnts = []

        ans = []

        # Count the unique numbers
        for n in uniq_nums:
            num_cnts.append(nums.count(n))

        # Collect top k frequent elements
        while len(ans) < k:
            # Get the indexes and the top element
            m = max(num_cnts)
            m_idx = num_cnts.index(m)
            freq = uniq_nums[m_idx]
            ans.append(freq)

            # Remove the items found
            num_cnts.remove(num_cnts[m_idx])
            uniq_nums.remove(uniq_nums[m_idx])

        return ans

    '''
    [Leetcode: Easy] (349) Intersection of Two Arrays, 6/8/2016
    Given two arrays, write a function to compute their intersection.

    Example:
    Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2].

    Note:
    Each element in the result must be unique.
    The result can be in any order.
    '''
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """

        if len(nums1) == 0 or len(nums2) == 0:
            return []

        # Simple solution with set operator
        return list(set(nums1) & set(nums2))

    '''
    [Leetcode: Easy] (165) Intersection of Two Arrays, 6/8/2016
    Compare two version numbers version1 and version2.
    If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.

    You may assume that the version strings are non-empty and contain only digits and the . character.
    The . character does not represent a decimal point and is used to separate number sequences.
    For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision
    of the second first-level revision.

    Here is an example of version numbers ordering:
    0.1 < 1.1 < 1.2 < 13.37
    '''
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        def adj_vlen(v, diff):
            for i in range(diff):
                v.append(0)
            return v

        v1 = version1.split('.')
        v2 = version2.split('.')

        # Example cases to consider
        # 1.3.2 > 1.2
        # 1.0 > 0.99
        # 2.42 < 2.8.0
        # 3 > 2.9.9.10
        # 1.3.2.0 == 1.3.2

        chk_len = max(len(v1), len(v2))
        len_diff = abs(len(v1) - len(v2))

        # Adjust the length of the shorter version
        if len(v1) > len(v2):
            v2 = adj_vlen(v2, len_diff)

        if len(v1) < len(v2):
            v1 = adj_vlen(v1, len_diff)

        # Check each digit to compare a version
        for i in range(chk_len):
            if int(v1[i]) > int(v2[i]):
                return 1
            elif int(v1[i]) < int(v2[i]):
                return -1
            else:
                pass

        return 0

    '''
    [Leetcode: Easy] (14) Longest Common Prefix, 6/11/2016
    Write a function to find the longest common prefix string amongst an array of strings.
    '''
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """

        if len(strs) == 0:
            return ""

        min_str_len = 2**32 - 1 # MAX_INT

        for s in strs:
            str_len = len(s)
            if str_len < min_str_len:
                min_str_len = str_len

        prefix_len = 0

        # Check ith digit at jth string
        for i in range(min_str_len):
            digit_strs = set()

            # Length of set would be 1 if all digits of strings are identical
            for j in range(len(strs)):
                digit_strs.add(strs[j][i])

            if len(digit_strs) == 1:
                prefix_len += 1
            else:
                break

        return strs[0][0:prefix_len]

    '''
    [Leetcode: Easy] (350) Intersection of Two Arrays II, 6/11/2016
    Given two arrays, write a function to compute their intersection.

    Example:
    Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].

    Note:
    Each element in the result should appear as many times as it shows in both arrays.
    The result can be in any order.
    '''
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """

        # Find (element:occurence) pairs
        def get_stats(nums):
            stat = dict()
            for n in nums:
                if n in stat:
                    stat[n] += 1
                else:
                    stat[n] = 1
            return stat

        if len(nums1) == 0 or len(nums2) == 0:
            return []

        ans = []
        n1_s = get_stats(nums1)
        n2_s = get_stats(nums2)

        intersect = list(set(n1_s) & set(n2_s))

        # Find how many times the intersection of arrays show up
        for i in intersect:
            ans += [i] * min(n1_s[i], n2_s[i])

        return ans

    '''
    [Leetcode: Easy] (102) Binary Tree Level Order Traversal, 6/15/2016
    Given a binary tree, return the level order traversal of its nodes' values.
    (ie, from left to right, level by level).

    For example:
    Given binary tree [3,9,20,null,null,15,7],
        3
       / \
      9  20
        /  \
       15   7
    return its level order traversal as:
    [
      [3],
      [9,20],
      [15,7]
    ]
    '''
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        def levelOrderHelper(root, depth, traversal):
            if root == None:
                return None

            # As a depth goes, read values to the same list
            if depth not in traversal:
                traversal[depth] = [root.val]
            else:
                traversal[depth] += [root.val]

            levelOrderHelper(root.left, depth+1, traversal)
            levelOrderHelper(root.right, depth+1, traversal)

        if root == None:
            return []

        traversal = {}
        levelOrderHelper(root, 0, traversal)

        return traversal.values()

    '''
    [Leetcode: Easy] (107) Binary Tree Level Order Traversal II, 6/15/2016
    Given a binary tree, return the bottom-up level order traversal of its nodes' values.
    (ie, from left to right, level by level from leaf to root).

    For example:
    Given binary tree [3,9,20,null,null,15,7],
        3
       / \
      9  20
        /  \
       15   7
    return its bottom-up level order traversal as:
    [
      [15,7],
      [9,20],
      [3]
    ]
    '''
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        def levelOrderBottomHelper(root, level, traversal):
            if root == None:
                return None

            if level not in traversal:
                traversal[level] = [root.val]
            else:
                traversal[level] += [root.val]

            levelOrderBottomHelper(root.left, level+1, traversal)
            levelOrderBottomHelper(root.right, level+1, traversal)

        if root == None:
            return []

        traversal = {}
        levelOrderBottomHelper(root, 0, traversal)

        # Return the result in a reverse way (Same approach with Leet102)
        return traversal.values()[::-1]

    '''
    [Leetcode: Easy] (357) Count Numbers with Unique Digits, 6/15/2016
    Given a non-negative integer n, count all numbers with unique digits, x, where 0 ≤ x < 10n.

    Example:
    Given n = 2, return 91. (The answer should be the total numbers in the range of 0 ≤ x < 100,
    excluding [11,22,33,44,55,66,77,88,99])
    '''
    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        """

        # uniq digits where n
        # n = 1:
        #   9 * 1 + 1 = 10
        # n = 2:
        #   first possible digits = 9
        #   next possible digits = 9
        #   9 * 9 + all_digits(n=1) = 81+10 = 91
        # n = 3:
        #   9 * 9 * 8 + all_digits(n=2) = 648 + 91 = 739
        # n = 4:
        #   9 * 9 * 8 * 7 + 739 = 5276, ...

        def nthUniqueDigits(n):
            s = 9
            for i in range(9, 10-n, -1):
                s *= i
            return s

        # If more than 9 digits, no number would have unique digits
        if n > 9:
            n = 9

        # Count 0 as a special case
        ans = 1
        while n > 0:
            ans += nthUniqueDigits(n)
            n -= 1

        return ans

    '''
    [Leetcode: Easy] 389. Find the Difference
    Given two strings s and t which consist of only lowercase letters.
    String t is generated by random shuffling string s and then add one more letter at a random position.
    Find the letter that was added in t.

    Example:
    Input:
    s = "abcd"
    t = "abcde"

    Output:
    e

    Explanation:
    'e' is the letter that was added.
    '''
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """

        # After all XOR, we have only one different character.
        res = 0x0
        for i in range(len(s)):
            res ^= ord(s[i]) ^ ord(t[i])

        return chr(res^ord(t[len(t)-1]))

    '''
    [Leetcode: Medium] (390) Elimination Game, 10/01/2016
    There is a list of sorted integers from 1 to n. Starting from left to right, remove the first number and every other number afterward until you reach the end of the list.
    Repeat the previous step again, but this time from right to left, remove the right most number and every other number from the remaining numbers.
    We keep repeating the steps again, alternating left to right and right to left, until a single number remains.
    Find the last number that remains starting with a list of length n.

    Example:
    Input:
    n = 9,
    1 2 3 4 5 6 7 8 9
    2 4 6 8
    2 6
    6

    Output:
    6
    '''
    def lastRemaining(self, n):
        """
        :type n: int
        :rtype: int
        """

        '''
        # The following looks time-excceeded due to heavy pop operation..!
        stack = range(1, n+1)
        iteration = 0

        while len(stack) > 1:
            numElement = len(stack)
            numRemoval = numElement/2 if numElement%2 == 0 else numElement/2 + 1
            if iteration % 2 == 0:
                for i in range(numRemoval):
                    stack.pop(i)
            else:
                for i in range(numRemoval):
                    stack.pop(-(i+1))
            iteration += 1

        return stack[0]
        '''

        '''
        # Another solution looks reasonable but time limit at 100000000!
        s = range(1, n+1)
        while len(s) > 1:
            s = s[1::2][::-1]

        return s[0]
        '''

        def isRemoveFront(i):
            return i % 2 == 0

        # k: number of items to be eliminated then n can be represented as:
        #    n = 2k - 1 where n is odd,
        #    n = 2k     where n is even
        # r = n - k: number of remaining items after removal
        # (s, e, diff): Sequences with common diffrence 'diff', starting from 's' to end 'e'

        s, e, diff = 1, n, 1
        k = (n+1)/2
        r = n-k
        iteration = 1

        while r > 0:
            # Eliminate items from the Front
            if isRemoveFront(iteration):
                e -= diff               # end would be the previous number of current sequence
                diff *= 2               # diff should be mutiplied to determine current range (s,e,diff)
                s = e - (r-1)*diff      # Get the smallest start bigger than previous s

            # Eliminate items from the Back
            else:
                s += diff               # start would be the next number of current sequence
                diff *= 2               # diff is multiplied by 2 at all times for next iteration
                e = s + (r-1)*diff      # Get the largest end less than previous e

            k = (r+1)/2
            r -= k
            iteration += 1

        # Ends up with s == e now
        return s