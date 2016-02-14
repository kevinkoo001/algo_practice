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