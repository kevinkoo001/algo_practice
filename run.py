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

def test_main(kind, prob):
    ct = CodeTest(kind)
    ct.proc_test(prob)

if __name__ == '__main__':
    # Change the problem set and the number
    test_main('leet', '304')