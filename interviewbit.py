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