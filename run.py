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

    def interviewbit_004(self):
        print self.interviewbit.maxSubArray([-2,1,-3,4,-1,2,1,-5,4])
        print self.interviewbit.maxSubArray([ -37, -450, -340, 79, -148, -306, -212, -373, -41, -260, 26, -295, -48, -84, -481, -225, -487, -484, -359, -485, -274, -465, -206, -90, -168, -488, -338, -192, -449, -156, -151, 28, -121, 93, 34, -152, -266, 12, -454, -41, -125, -311, -388, -215, -262, -182, -449, -30, -336, -330, -54, -349, 27, 18, -100, -215, -76, -275, -170, -48, -404, -157, 6, -21, -456, 0, -203, 75, -111, -87, -54, -264, -489, -340, -288, -291, -115, -437, -22, -454, -467, -161, -499, -313, -410, -276, -90, -125, -108, -367, -400, -74, 47, -401, -166, -419, 82, -437, 28, 40, 25, -121, 23, -433, 43, 34, -144, -51, -309, -147, -310, -172, -33, 72, 83, -423, -259, -373, -54, -450, -318, -357, -200, -179, -330, -486, -55, -427, -474, -163, 23, -437, -323, -394, -39, 52, -416, -498, -174, -405, -130, -223, -388, -452, -423, -144, -317, -405, -335, -470, -304, -441, -246, -415, 16, 31, -269, -407, -412, 54, -416, 84, -241, 14, -296, -234, -216, -384, -106, -361, -317, -182, -344, -441, -49, -281, 1, 53, -407, 35, -31, -120, -488, -198, 77, -300, -494, 41, -315, -232, -11, -221, -484, 60, -342, -402, -352, 94, -473, -269, -302, -439, -467, -427, -9, -66, -379, -278, -34, -95, -404, -491, -260, -392, -5, -177, -268, -362, -299, -207, -235, -127, -305, -448, -310, -392, -338, -382, -73, -225, -22, 81, -267, 88, -445, -89, -247, -114, -368, -429, -280, -110, -141, -296, -257, -387, 9, -344, -96, -451, -365, -178, -149, -233, -487, -414, 47, -431, -19, -312, 42, -188, -370, -282, 27, -336, -255, -474, -71, -412, -28, -224, -147, -259, -323, -406, -368, -112, 88, -336, -475, -101, -375, -317, -166, -493, -261, -38, -225, -385, -452, 24, -355, -57, -385, -338, -129, -83, -384, -278, -162, 25, -232, -324, -334, -18, -299, -454, -111, -265, -493, -210, -349, -17, -211, -347, -211, -300, -308, -257, -372, -48, -157, -91, -430, -53, -369, -448, -158, -422, -459, -349, 7, -275, -340, 2, -103, -268, -499, -64, -46, 28, -93, -226, -108, -78, -141, -66, -481, -139, -402, -448, -383, -260, 17, 9, -401, -327, -453, -334, -12, -170, -221, -22, -347, -53, -323, 31, -87, -406, 100, 76, -384, -247, -440, -180, -289, -125, -219, -459, 87, -401, -438, -308, -369, -308, -123, -186, 38, 37, -405, -137, -291, -138, -49, -119, -342, -376, -469, 2, -187, -43, 41, -304, -71, -101, -26, 36, -96, -72, -287, -484, -331, -139, -239, -248, -104, -51, -225, -363, -100, -365, 94, -288, -324, -36, -216, 9, -387, -234, -204, -363, -304, -221, -8, -10, -375, -136, 95, -340, -382, -232, -96, -298, -211, 91, -67, -49, -189, -43, 73, -254, -346, -498, -443, -268, -207, -143, -297, -332, -269, -187, -91, -11, 57, -480, -434, -218, 76, -495, -245, 57, -250, -316, -192, -12, -157, -475, 26, -44, -395, 17, 11, -389, 91, -433, -75, -133, 64, -100, -383, -266, -158, -249, 56, -140, -389, -275, -470, -66, -42, -275, 7, -170, -71, -154, -482, -364, -379, -19, -121, 42, 45, -202, -26, -51, -68, -256, -407, 65, -262, 18, -226, -137, -31, 87, 11, -109, -85, -336, -80, -452, -160, -37, -477, -125, -90, -486, 21, -401, 65, -289, -72, 29, -430, -210, -491, -64, 64, -149, -311, 23, -212, -314, -45, -498, -319, 88, -298, -189, 82, 79, 25, -380, -12, -388, 18, 76, -463, -387, -456, -299, -449, -271, -28, -9, 42, -247, -75, 7, -103, -212, -52, -246, 35, -247, -275, -216, -127, -63, 29, -166, -482, 16, -31, -406, -457, -394, -296, -138, -109, 84, -59, -117, -182, -20, 4, -260, -420, -286, -129, -131, -402, -87, -199, -87, -362, 45, -39, -250, -47, -65, -276, 99, -411, -266, -434, -348, -61, -283, -62, -237, -59, -151, -333, -108, -309, -322, -394, -44, -222, -455, -464, -141, -7, -358, -314, 92, -208, -58, -357, -448, -397, -9, -148, -141, -495, 38, 13, -465, -20, -113, 63, -392, -218, -184, 5, -219, -458, -248, -444, 82, -74, -393, -63, -15, -175, -429, -457, -288, -381, -344, -329, -326, -117, -108, -432, -418, 70, -3, 87, -112, 91, -451, -54, -134, -288, -168, -64, -437, -481, -390, -96, -439, -153, -50, -76, -196, -176, -159, -330, -462, -16, -482, 52, -43, -485, -500, -326, -2, -271, 8, -260, -18, -420, -316, -291, 83, -489, -324, -241, -107, 8, -457, -221, -302, 5, -399, -218, -91, -113, -302, -146, -473, -402, -2, -494, -120, -88, -417, -121, -249, -473, -402, -154, -255, -98, 45, -492, -423, -315, 19, -188, -62, -301, -108, -251, -467, -332, 20, -349, 88, -48, -365, 58, -331, -372, -205, -34, 78, -393, 39, -232, -418, -44, -155, -92, -140, -275, 20, -130, -65, 69, -430, -244, -222, -101, -366, -149, -159, 41, -151, -49, -36, -206, -153, -152, -140, -206, -142, 55, -477, -481, 17, 34, 88, -212, -84, -172, -390, -260, -198, 65, -425, -146, -38, 29, -488, -159, -478, -194, -173, -194, -253, 90, -76, -153, -397, 84, 50, -432, 86, -435, -218, -343, -161, -353, -171, -466, -410, -42, -31, 86, -281, -132, -6, 51, -474, -495, -324, 37, -402, -422, -309, -280, -223, -38, -352, -270, -464, -403, 67, -179, -209, 7, -314, -308, -182, 71, -251, -330, -437, -362, -233, 2, -41, -360, -228 ])
        print self.interviewbit.maxSubArray([ -120, -202, -293, -60, -261, -67, 10, 82, -334, -393, -428, -182, -138, -167, -465, -347, -39, -51, -61, -491, -216, -36, -281, -361, -271, -368, -122, -114, -53, -488, -327, -182, -221, -381, -431, -161, -59, -494, -406, -298, -268, -425, -88, -320, -371, -5, 36, 89, -194, -140, -278, -65, -38, -144, -407, -235, -426, -219, 62, -299, 1, -454, -247, -146, 24, 2, -59, -389, -77, -19, -311, 18, -442, -186, -334, 41, -84, 21, -100, 65, -491, 94, -346, -412, -371, 89, -56, -365, -249, -454, -226, -473, 91, -412, -30, -248, -36, -95, -395, -74, -432, 47, -259, -474, -409, -429, -215, -102, -63, 80, 65, 63, -452, -462, -449, 87, -319, -156, -82, 30, -102, 68, -472, -463, -212, -267, -302, -471, -245, -165, 43, -288, -379, -243, 35, -288, 62, 23, -444, -91, -24, -110, -28, -305, -81, -169, -348, -184, 79, -262, 13, -459, -345, 70, -24, -343, -308, -123, -310, -239, 83, -127, -482, -179, -11, -60, 35, -107, -389, -427, -210, -238, -184, 90, -211, -250, -147, -272, 43, -99, 87, -267, -270, -432, -272, -26, -327, -409, -353, -475, -210, -14, -145, -164, -300, -327, -138, -408, -421, -26, -375, -263, 7, -201, -22, -402, -241, 67, -334, -452, -367, -284, -95, -122, -444, -456, -152, 25, 21, 61, -320, -87, 98, 16, -124, -299, -415, -273, -200, -146, -437, -457, 75, 84, -233, -54, -292, -319, -99, -28, -97, -435, -479, -255, -234, -447, -157, 82, -450, 86, -478, -58, 9, -500, -87, 29, -286, -378, -466, 88, -366, -425, -38, -134, -184, 32, -13, -263, -371, -246, 33, -41, -192, -14, -311, -478, -374, -186, -353, -334, -265, -169, -418, 63, 77, 77, -197, -211, -276, -190, -68, -184, -185, -235, -31, -465, -297, -277, -456, -181, -219, -329, 40, -341, -476, 28, -313, -78, -165, -310, -496, -450, -318, -483, -22, -84, 83, -185, -140, -62, -114, -141, -189, -395, -63, -359, 26, -318, 86, -449, -419, -2, 81, -326, -339, -56, -123, 10, -463, 41, -458, -409, -314, -125, -495, -256, -388, 75, 40, -37, -449, -485, -487, -376, -262, 57, -321, -364, -246, -330, -36, -473, -482, -94, -63, -414, -159, -200, -13, -405, -268, -455, -293, -298, -416, -222, -207, -473, -377, -167, 56, -488, -447, -206, -215, -176, 76, -304, -163, -28, -210, -18, -484, 45, 10, 79, -441, -197, -16, -145, -422, -124, 79, -464, -60, -214, -457, -400, -36, 47, 8, -151, -489, -327, 85, -297, -395, -258, -31, -56, -500, -61, -18, -474, -426, -162, -79, 25, -361, -88, -241, -225, -367, -440, -200, 38, -248, -429, -284, -23, 19, -220, -105, -81, -269, -488, -204, -28, -138, 39, -389, 40, -263, -297, -400, -158, -310, -270, -107, -336, -164, 36, 11, -192, -359, -136, -230, -410, -66, 67, -396, -146, -158, -264, -13, -15, -425, 58, -25, -241, 85, -82, -49, -150, -37, -493, -284, -107, 93, -183, -60, -261, -310, -380 ])
        print self.interviewbit.maxSubArray([-500])

def test_main(kind, prob):
    ct = CodeTest(kind)
    ct.proc_test(prob)

if __name__ == '__main__':
    test_main('interviewbit', '004')