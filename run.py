import interviewbit
import leetcode

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

def test_main(kind, prob):
    ct = CodeTest(kind)
    ct.proc_test(prob)

if __name__ == '__main__':
    test_main('leet', '004')