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