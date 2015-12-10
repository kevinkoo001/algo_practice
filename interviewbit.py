class Solution():
    def __init__(self):
        pass

    '''
    [Interviewbit: Level 2 Arrays] PRETTYPRINT, 12/08/2015

    Print out like the following
    5 5 5 5 5 5 5 5 5
    5 4 4 4 4 4 4 4 5
    5 4 3 3 3 3 3 4 5
    5 4 3 2 2 2 3 4 5
    5 4 3 2 1 2 3 4 5
    5 4 3 2 2 2 3 4 5
    5 4 3 3 3 3 3 4 5
    5 4 4 4 4 4 4 4 5
    5 5 5 5 5 5 5 5 5
    '''
    def prettyPrint(self, A):
        result = []
        perimeter = [A] * A

        start = A
        for i in range(A):
            for j in range(i, A):
                perimeter[j] = start
            start -= 1
            symmetric = [x for x in reversed(perimeter)]
            result.append(perimeter + symmetric[1:len(symmetric)])

        mirror = [x for x in reversed(result)]
        result.extend(mirror[1:len(mirror)])

        return result

    '''
    [Interviewbit: Level 2 Arrays] MAXSET, 12/08/2015

    Find out the maximum sub-array of non negative numbers from an array.
    The sub-array should be continuous. That is, a sub-array created by choosing
    the second and fourth element and skipping the third element is invalid.
    Ex)
        A : [1, 2, 5, -7, 2, 3]
        The two sub-arrays are [1, 2, 5] [2, 3].
        The answer is [1, 2, 5] as its sum is larger than [2, 3]
    '''
    def maxset(self, A):
        walls = []
        prev = -1
        for i in range(len(A)):
            if A[i] < 0:
                walls.append((prev, i))
                prev = i

        walls.append((prev, len(A)))
        left, right, max = -1, -1, -1

        for i, j in walls:
            value = sum(A[i+1:j])
            if value > max:
                max = value
                left, right = i+1, j

        if left > 0 or right > 0:
            return A[left:right]
        else:
            return []

    '''
    [Interviewbit: Level 2 Arrays] REACH, 12/09/2015

    You are given a sequence of points and the order in which you need to cover the points.
    Give the minimum number of steps in which you can achieve it. You start from the first point.
        # @param X : list of integers
        # @param Y : list of integers
        # Points are represented by (X[i], Y[i])
        # @return an integer
        Ex) [(0, 0), (1, 1), (1, 2)], 2
    '''
    def coverPoints(self, X, Y):
        def simple_distance(A, B):
            x1, y1 = A
            x2, y2 = B

            '''
            if y1 == y2:
                return abs(x1 - x2)
            elif x1 == x2:
                return abs(y1 - y2)
            else:
                return 0
            '''

            return abs(x1 - x2) if y1 == y2 else abs(y1 - y2)

        # Extract all points
        T = zip(X, Y)

        # Compute the shortest path for each step from point A to B
        total = 0
        for A, B in [(T[x], T[x+1]) for x in range(len(T)-1)]:
            x1, y1 = A
            x2, y2 = B
            if abs(x1 - x2) == 0 or abs(y1 - y2) == 0:
                total += simple_distance(A, B)
            else:
                # fast_move is defined as moving diagonal path close to B from A
                fast_move = min(abs(x2 - x1), abs(y2 - y1))
                total += fast_move
                if x1 < x2 and y1 < y2:
                    total += simple_distance((x1 + fast_move, y1 + fast_move), B)
                elif x1 < x2 and y1 > y2:
                    total += simple_distance((x1 + fast_move, y1 - fast_move), B)
                elif x1 > x2 and y1 < y2:
                    total += simple_distance((x1 - fast_move, y1 + fast_move), B)
                else:   # x1 > x2 and y1 > y2
                    total += simple_distance((x1 - fast_move, y1 - fast_move), B)

        return total

    '''
    [Interviewbit: Level 2 Arrays] MAXSUM, 12/10/2015

    Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
    Ex) Given the array [-2,1,-3,4,-1,2,1,-5,4],
        the contiguous subarray [4,-1,2,1] has the largest sum = 6.
    '''
    # @param A : tuple of integers
    # @return an integer
    def maxSubArray(self, A):

        # Kadane's Algorithm: O(N)
        ans, s = 0, 0
        for i in range(1, len(A)):
            if s + A[i] > 0:
                s += A[i]
            else:
                s = 0
            ans = max(ans, s)

        '''
        # Naive approach - Brute Force: O(N^2)
        max = -99999999
        for i in range(len(A)):
            for j in range(i + 1, len(A) + 1):
                tmp = sum(A[i:j])
                if tmp > max:
                    max = tmp
                    #ans = A[i:j]
        return max

        # Divide and Conquer: O(N log N) - need to be fixed
        if len(A) == 1:
            return A[0]

        m = len(A)/2
        left_mss = self.maxSubArray(A[0:m])
        right_mss = self.maxSubArray(A[m:])

        left_sum = -99999999
        right_sum = -99999999

        right_sum = max(right_sum, sum(A[m:]))
        left_sum = max(left_sum, sum(A[0:m]))

        ans = max(left_mss, right_mss)
        return max(ans, left_sum + right_sum)
        '''

        return ans if ans > 0 else 0

