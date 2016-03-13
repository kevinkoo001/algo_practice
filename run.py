import interviewbit
import leetcode
import basicds

'''
Reference sites:
    interviewbit.com
    leetcode.com
    geeksforgeeks.org
'''

class CodeTest(object):
    # prob_num consists of three-digit numbers
    def __init__(self, test_set):
        self.test_set = test_set
        if test_set == 'leet':
            self.leet = leetcode.Solution()
        elif test_set == 'interviewbit':
            self.interviewbit = interviewbit.Solution()
        else:
            pass

    # Define problem test sets for 'leetcode'
    def proc_test(self, prob_num):
        test_name = self.test_set + '_' + str(prob_num)
        testset = getattr(self, test_name, lambda: "nothing")
        return testset()

    def leet_001(self):
        print self.leet.twoSum([2,7,11,15], 9)
        print self.leet.twoSum([2,7,7,8], 14)
        print self.leet.twoSum([2,7,11,2], 18)
        print self.leet.twoSum([3,2,4], 6)

    def leet_002(self):
        n = 12
        print self.leet.pick_largest(n)

    def leet_003(self):
        A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        B = self.leet.performOps(A)
        for i in xrange(len(B)):
            for j in xrange(len(B[i])):
                print B[i][j],
        '''
        while n > 0:
            (L, R) = leet.pick_largest(n)
        '''

    def leet_004(self):
        print self.leet.isUgly(0)

    def leet_217(self):
        print self.leet.containsDuplicate([1,2,4,5,6])
        print self.leet.containsDuplicate([1,2,4,4,6])
        print self.leet.containsDuplicate([])

    def leet_219(self):
        print self.leet.containsNearbyDuplicate([1,2,3,5,4,2,1], 3)

    # Define problem test sets for 'interviewbit'
    def interviewbit_001(self):
        local_test = [[1, 2, 5, -7, 2, 3], [1, 2, 5, 7, 2, 3],
                      [-1, -2, -5, -7, -2, -3], [0, 0, -1, 0],
                      [-846930886, -1714636915, 424238335, -1649760492]]

        for i in range(len(local_test)):
            print self.interviewbit.maxset(local_test[i])

    def interviewbit_002(self):
        result = self.interviewbit.prettyPrint(7)
        for i in range(len(result)):
            for j in range(len(result[0])):
                print result[i][j],
            print '\n',

    def interviewbit_003(self):
        print self.interviewbit.coverPoints([0,1,1], [0,1,2])

    def interviewbit_004(self):
        print self.interviewbit.maxSubArray([-2,1,-3,4,-1,2,1,-5,4])
        print self.interviewbit.maxSubArray([-500])

    def leet_234(self):
        x1 = basicds.ListNode(1)
        x2 = basicds.ListNode(2)
        x3 = basicds.ListNode(3)
        x4 = basicds.ListNode(4)
        x5 = basicds.ListNode(4)
        x6 = basicds.ListNode(3)
        x7 = basicds.ListNode(2)
        x8 = basicds.ListNode(1)
        x1.next = x2
        x2.next = x3
        x3.next = x4
        x4.next = x5
        x5.next = x6
        x6.next = x7
        x7.next = x8
        head = x1
        print self.leet.isPalindrome(head)

    def leet_049(self):
        print self.leet.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])

    def leet_226(self):
        root = basicds.TreeNode(4,
            basicds.TreeNode(7,
                 basicds.TreeNode(2,
                      basicds.TreeNode(9, None, None),
                      None),
                 basicds.TreeNode(6, None, None)),
            basicds.TreeNode(3,
                 basicds.TreeNode(1,
                      basicds.TreeNode(8, None, None),
                      basicds.TreeNode(5, None, None)),
                 None))

        print '\nBefore:'
        basicds.levelorder(root)
        self.leet.invertTree(root)
        print '\nAfter:'
        basicds.levelorder(root)

    def leet_304(self):
        matrix = [[3, 0, 1, 4, 2],
              [5, 6, 3, 2, 1],
              [1, 2, 0, 1, 5],
              [4, 1, 0, 1, 7],
              [1, 0, 3, 0, 5]]

        nm = self.leet.NumMatrix(matrix)
        print nm.sumRegion(2, 1, 4, 3)
        print nm.sumRegion(1, 1, 2, 2)
        print nm.sumRegion(1, 2, 2, 4)

    def leet_292(self):
        print self.leet.canWinNim(12)

    def leet_328(self):
        head = basicds.ListNode(1)
        n2 = basicds.ListNode(2)
        n3 = basicds.ListNode(3)
        n4 = basicds.ListNode(4)
        n5 = basicds.ListNode(5)
        n6 = basicds.ListNode(6)
        head.next = n2
        n2.next = n3
        n3.next = n4
        n4.next = n5
        n5.next = n6

        basicds.printList(head)
        self.leet.oddEvenList(head)
        basicds.printList(head)

    def leet_206(self):
        x1 = basicds.ListNode(1)
        x2 = basicds.ListNode(2)
        x3 = basicds.ListNode(3)
        x4 = basicds.ListNode(4)
        x5 = basicds.ListNode(5)
        x1.next = x2
        x2.next = x3
        x3.next = x4
        x4.next = x5
        x5.next = None

        basicds.printList(x1)
        new_head = self.leet.reverseList(x1)
        basicds.printList(new_head)

    def leet_231(self):
        print self.leet.isPowerOfTwo(10)
        print self.leet.isPowerOfTwo(16)
        print self.leet.isPowerOfTwo(2**20)
        print self.leet.isPowerOfTwo(2**15 - 1)

    def leet_326(self):
        print self.leet.isPowerOfThree(10)
        print self.leet.isPowerOfThree(27)
        print self.leet.isPowerOfThree(3**10)
        print self.leet.isPowerOfThree(3**20 + 1)

    def leet_190(self):
        print self.leet.reverseBits(43261596)

    def leet_007(self):
        print self.leet.reverse(1563847412)
        print self.leet.reverse(9654534)
        print self.leet.reverse(-1234567)

    def leet_100(self):
        p1 = basicds.TreeNode(4,
            basicds.TreeNode(7,
                 basicds.TreeNode(2,
                      basicds.TreeNode(9, None, None),
                      None),
                 basicds.TreeNode(6, None, None)),
            basicds.TreeNode(3,
                 basicds.TreeNode(1,
                      basicds.TreeNode(8, None, None),
                      basicds.TreeNode(5, None, None)),
                 None))

        q1 = basicds.TreeNode(4,
            basicds.TreeNode(7,
                 basicds.TreeNode(2,
                      basicds.TreeNode(9, None, None),
                      None),
                 basicds.TreeNode(6, None, None)),
            basicds.TreeNode(3,
                 basicds.TreeNode(1,
                      basicds.TreeNode(4, None, None),
                      basicds.TreeNode(5, None, None)),
                 None))

        print self.leet.isSameTree(p1, q1)

        p2 = basicds.TreeNode(4,
            basicds.TreeNode(7,
                 basicds.TreeNode(2,
                      basicds.TreeNode(9, None, None),
                      None),
                 basicds.TreeNode(6, None, None)))

        q2 = basicds.TreeNode(4,
            basicds.TreeNode(7,
                 basicds.TreeNode(2,
                      basicds.TreeNode(9, None, None),
                      None)))

        print self.leet.isSameTree(p2, q2)

    def leet_009(self):
        print self.leet.isPalindrome(12344321)
        print self.leet.isPalindrome(123454321)
        print self.leet.isPalindrome(123451)

    def leet_140(self):
        print self.leet.wordBreak("catsanddog", ["cat", "cats", "and", "sand", "dog", "catsand", "catsanddog"])

    def leet_057(self):
        def setA():
            r1 = basicds.Interval(s=1, e=2)
            r2 = basicds.Interval(s=3, e=5)
            r3 = basicds.Interval(s=6, e=7)
            r4 = basicds.Interval(s=8, e=10)
            r5 = basicds.Interval(s=12, e=16)
            range_list = list()
            range_list.append(r1)
            range_list.append(r2)
            range_list.append(r3)
            range_list.append(r4)
            range_list.append(r5)
            return range_list

        def setB():
            range_list = list()
            r1 = basicds.Interval(s=2, e=5)
            r2 = basicds.Interval(s=6, e=7)
            r3 = basicds.Interval(s=8, e=9)
            range_list.append(r1)
            range_list.append(r2)
            range_list.append(r3)
            return range_list

        def setC():
            range_list = list()
            r1 = basicds.Interval(s=0, e=1)
            r2 = basicds.Interval(s=5, e=5)
            r3 = basicds.Interval(s=6, e=7)
            r4 = basicds.Interval(s=9, e=11)
            range_list.append(r1)
            range_list.append(r2)
            range_list.append(r3)
            range_list.append(r4)
            return range_list

        result = self.leet.insert(setA(), basicds.Interval(s=4, e=9))
        #result = self.leet.insert(setB(), basicds.Interval(s=0, e=1))
        #esult = self.leet.insert(setC(), basicds.Interval(s=50, e=50))
        for interval in result:
            print (interval.start, interval.end),

    def leet_111(self):
        root = basicds.TreeNode(4, basicds.TreeNode(7, basicds.TreeNode(2, None), None), basicds.TreeNode(1, None, None))
        print self.leet.minDepth(root)

    def leet_101(self):
        asymmetric_root = basicds.TreeNode(4, basicds.TreeNode(7, basicds.TreeNode(2, None), None), basicds.TreeNode(1, None, None))
        symmetric_root = basicds.TreeNode(4, basicds.TreeNode(7, basicds.TreeNode(2, None), None), basicds.TreeNode(7, None, basicds.TreeNode(2)))
        print self.leet.isSymmetric(symmetric_root)
        print self.leet.isSymmetric(asymmetric_root)

    def leet_205(self):
        print self.leet.isIsomorphic('aab', 'bbc')
        print self.leet.isIsomorphic('aa', 'bb')
        print self.leet.isIsomorphic('ab', 'aa')

    def leet_331(self):
        print self.leet.isValidSerialization("9,#,92,#,#")
        print self.leet.isValidSerialization("9,3,4,#,#,1,#,#,2,#,6,#,#")
        print self.leet.isValidSerialization("#")
        print self.leet.isValidSerialization("9,#,#,1")
        print self.leet.isValidSerialization("1, #")

    def leet_131(self):
        print self.leet.partition("aab")
        print self.leet.partition("aabcacba")


    def leet_083(self):
        node1 = basicds.ListNode(1)
        node2 = basicds.ListNode(1)
        node3 = basicds.ListNode(2)
        node4 = basicds.ListNode(3)
        node1.next = node2
        node2.next = node3
        node3.next = node4
        head = node1
        basicds.printList(head)
        self.leet.deleteDuplicates(head)
        basicds.printList(head)

    def leet_027(self):
        ar = [1,3,4,5,3,2,1,3,6,7,3,4]
        target = 3
        print "Original length of array: %d (%s)" % (len(ar), ar)
        print "Removing target element: %d" % target
        print "Array length after element removal: %d (%s)" % (self.leet.removeElement(ar, target), ar)

    def leet_204(self):
        print self.leet.countPrimes(100000)

    def leet_046(self):
        self.leet.permute([1,2,3,4])

    def leet_319(self):
        print self.leet.bulbSwitch(1000)

    def leet_160(self):
        node1 = basicds.ListNode(1)
        node2 = basicds.ListNode(1)
        node3 = basicds.ListNode(2)
        node4 = basicds.ListNode(3)
        node5 = basicds.ListNode(10)
        node1.next = node2
        node2.next = node3
        node3.next = node4
        node5.next = node4
        headA = node1
        headB = node5

        # intersection of the node -> node4 (node4.val = 3)
        intersection = self.leet.getIntersectionNode(headA, headB)
        print intersection.val

    def leet_203(self):
        node1 = basicds.ListNode(1)
        node2 = basicds.ListNode(1)
        node3 = basicds.ListNode(2)
        node4 = basicds.ListNode(3)
        node5 = basicds.ListNode(10)
        node1.next = node2
        node2.next = node3
        node3.next = node4
        node4.next = node5
        head = node1
        basicds.printList(node1)
        self.leet.removeElements(head, 3)
        basicds.printList(node1)

    def leet_141(self):
        node1 = basicds.ListNode(1)
        node2 = basicds.ListNode(1)
        node3 = basicds.ListNode(2)
        node4 = basicds.ListNode(3)
        node1.next = node2
        node2.next = node3
        node3.next = node4
        node4.next = node2
        head = node1
        print self.leet.hasCycle(head)

    def leet_142(self):
        node1 = basicds.ListNode(1)
        node2 = basicds.ListNode(2)
        node3 = basicds.ListNode(3)
        node4 = basicds.ListNode(4)
        node5 = basicds.ListNode(5)
        node1.next = node2
        node2.next = node3
        node3.next = node4
        node4.next = node5
        node5.next = node3
        head = node1
        print self.leet.detectCycle(head).val

    def leet_092(self):
        node1 = basicds.ListNode(1)
        node2 = basicds.ListNode(2)
        node3 = basicds.ListNode(3)
        node4 = basicds.ListNode(4)
        node5 = basicds.ListNode(5)
        node1.next = node2
        node2.next = node3
        node3.next = node4
        node4.next = node5
        head = node1
        basicds.printList(head)
        self.leet.reverseBetween(head, 2, 4)
        basicds.printList(head)

    def leet_303(self):
        arr = [8,7,4,3,2,1,5]
        print self.leet.NumArray(arr).sumRange(2,5)

    def leet_198(self):
        houses = [44, 22, 33, 77, 99, 11, 66]
        print self.leet.rob(houses)

    def leet_136(self):
        nums = [2,7,8,1,2,7,8]
        print self.leet.singleNumber(nums)

    def leet_137(self):
        nums = [2,4,2,2,7,7,7]
        print self.leet.singleNumber2(nums)

    def leet_260(self):
        nums = [1, 2, 1, 3, 2, 5]
        print self.leet.singleNumber3(nums)

    def leet_125(self):
        str = 'abcdedcba'
        print self.leet.isPalindrome2(str)

    def leet_088(self):
        nums1 = [1,4,5,7,9,0,0,0,0]
        nums2 = [2,6,8,10]
        self.leet.merge(nums1, 5, nums2, 4)
        print nums1

    def leet_118(self):
        print self.leet.generatePascal(8)

    def leet_067(self):
        print self.leet.addBinary('0', '0')
        print self.leet.addBinary('101', '110')
        print self.leet.addBinary('1001', '11110')

    def leet_026(self):
        nums = [1,1,2,3,3,4,5,5,5,5,6,9,10,10,11]
        pos = self.leet.removeDuplicates(nums)
        print nums[0:pos]

    def leet_020(self):
        print self.leet.isValid('(){}[()]')
        print self.leet.isValid(']')
        print self.leet.isValid('(((())))[{}]{[]}')
        print self.leet.isValid('(((()}]()')

def test_main(kind, prob):
    ct = CodeTest(kind)
    ct.proc_test(prob)

if __name__ == '__main__':
    # Change the problem set and the number
    test_main('leet', '020')