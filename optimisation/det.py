from time import time


def find_determinant(matrix,**kwargs):
    def power(base,exp):
        ans = 1
        for i in range(exp):
            ans *= base
        return ans
    
    def power_neg_one(exp):
        if not exp%2:
            return 1
        return -1
    
    def verify_square(matrix):
        size = len(matrix)
        for row in matrix:
            if len(row) != size:
                return False
        return True
    
    def get_submatrix(matrix,ignore_row,ignore_column):
        sub_matrix = []
        i = 0
        while i < len(matrix):
            row = []
            if i == ignore_row:
                i += 1
                continue
            j = 0
            while j < len(matrix):
                if j == ignore_column:
                    j += 1
                    continue
                row.append(matrix[i][j])
                j += 1
            sub_matrix.append(row)
            i += 1
        return sub_matrix

    start = kwargs.get("start")
    if start:
        if not verify_square(matrix):
            raise Exception("must be square matrix")
    if len(matrix) == 2:
        return ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))
    i = 0
    det = 0
    while i < len(matrix):
        if matrix[0][i] == 0:
            i += 1
            continue
        det += matrix[0][i] * power_neg_one(i) * find_determinant(get_submatrix(matrix,0,i),start = False)
        i += 1
    return det

matrix = [
    [0,2,3,4,3.2],
    [1,3,8,2.3,1.2],
    [2,4,5,1,3.4],
    [1.5,8.9,5.6,3.5,6.8],
    [5.4,5.6,3.5,3.1,1.4]
]
start = time()
print(find_determinant(matrix),time()-start)