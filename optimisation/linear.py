from cmath import inf
import copy

# a linear programming problem must be formulated with all inequalities being of the form variables less than or equal to constant
class LinearProgramming():
    def __init__(self):
        self.constraints = []
        self.my_equation = []
        self.bounds = []
    
    # takes a list of tuples in which the first item is a list of the coefficients and the second is a value
    def add_constraint(self,constraints):
        for constraint in constraints:
            self.constraints.append(constraint)

    # takes a list of the coefficients of the equation
    def set_equation(self,equation):
        self.my_equation = equation

    # takes a list of the individual lower bound of each variable
    def set_bounds(self,bounds):
        self.bounds = bounds
    
    # returns a list of the vertices of the constraint problem with each being a list of the values of all variables
    def get_vertices(self):
        def find_determinant(matrix,**kwargs):
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
        
        def check_vertex(vertex):
            def dot_product(a,b):
                ans = 0
                for i in range(len(a)):
                    ans += (a[i] * b[i])
                return ans

            for constraint in self.constraints:
                if dot_product(constraint[0],vertex) > constraint[1]:
                    return False
            return True

        def dot_product(a,b):
            ans = 0
            for i in range(len(a)):
                ans += (a[i] * b[i])
            return ans
        
        vertices = []
        fixed = []
        ans = []
        for i in range(len(self.constraints)):
            ans.append(self.bounds[i])
        if check_vertex(ans):
            vertices.append(ans)
        for constraint in self.constraints:
            fixed.append(copy.deepcopy(constraint[0]))
        result = []
        for constraint in self.constraints:
            result.append(copy.deepcopy(constraint[1]))
        matrix = copy.deepcopy(fixed)
        real_det = find_determinant(matrix)
        if real_det == 0:
            raise Exception("the system of inequalities is not independent")
        answer = []
        main = copy.deepcopy(matrix)
        for i in range(len(self.constraints)):
            j = 0
            matrix = copy.deepcopy(main)
            for row in matrix:
                row[i] = copy.deepcopy(result[j])
                j += 1
            answer.append(find_determinant(matrix)/real_det)
        vertices.append(answer)
        for i in range(len(self.constraints)):
            answer = []
            for j in range(len(self.constraints)):
                if i != j:
                    answer.append(copy.deepcopy(self.bounds[j]))
                else:
                    answer.append(0)
            for j in range(len(self.constraints)):
                if fixed[j][i] != 0:
                    answer[i] = (result[j] - dot_product(fixed[j],answer))/fixed[j][i]
                    if check_vertex(answer):
                        vertices.append(copy.deepcopy(answer))
                    answer[i] = 0
        return vertices

    # returns a list of the values of the variables that will maximize the result of the equation
    def maximize(self):
        def dot_product(a,b):
            ans = 0
            for i in range(len(a)):
                ans += (a[i] * b[i])
            return ans

        vertices = self.get_vertices()
        maximum = 0
        ans = []
        for vertex in vertices:
            result = dot_product(vertex,self.my_equation)
            if result > maximum:
                maximum = result
                ans = vertex
        return ans

    # returns a list of the values of the variables that will minimize the result of the equation
    def minimize(self):
        def dot_product(a,b):
            ans = 0
            for i in range(len(a)):
                ans += (a[i] * b[i])
            return ans

        vertices = self.get_vertices()
        minimum = inf
        ans = []
        for vertex in vertices:
            result = dot_product(vertex,self.my_equation)
            if result < minimum:
                minimum = result
                ans = vertex
        return ans
    
    # returns a list of the values of the variables that will maximize the result of the equation and only integer values
    def maximize_int(self):
        def dot_product(a,b):
            ans = 0
            for i in range(len(a)):
                ans += (a[i] * b[i])
            return ans
        
        def check_vertex(vertex):
            for constraint in self.constraints:
                if dot_product(constraint[0],vertex) > constraint[1]:
                    return False
            return True

        vertices = self.get_vertices()
        maximum = 0
        ans = []
        i = 0
        while i < len(vertices):
            changed = False
            for j in range(len(vertices[i])):
                if vertices[i][j] == int(vertices[i][j]):
                    pass
                else:
                    changed = True
                    temp = copy.deepcopy(vertices[i])
                    temp[j] = int(temp[j])
                    if check_vertex(temp):
                        if temp not in vertices:
                            vertices.append(copy.deepcopy(temp))
                    temp[j] += 1
                    if check_vertex(temp):
                        if temp not in vertices:
                            vertices.append(copy.deepcopy(temp))
            if changed:
                vertices.pop(i)
                continue
            i += 1
        for vertex in vertices:
            result = dot_product(vertex,self.my_equation)
            if result > maximum:
                maximum = result
                ans = vertex
        return ans

    # returns a list of the values of the variables that will minimize the result of the equation and only integer values
    def minimize_int(self):
        def dot_product(a,b):
            ans = 0
            for i in range(len(a)):
                ans += (a[i] * b[i])
            return ans
        
        def check_vertex(vertex):
            for constraint in self.constraints:
                if dot_product(constraint[0],vertex) > constraint[1]:
                    return False
            return True

        vertices = self.get_vertices()
        minimum = inf
        ans = []
        i = 0
        while i < len(vertices):
            changed = False
            for j in range(len(vertices[i])):
                if vertices[i][j] == int(vertices[i][j]):
                    pass
                else:
                    changed = True
                    temp = copy.deepcopy(vertices[i])
                    temp[j] = int(temp[j])
                    if check_vertex(temp):
                        if temp not in vertices:
                            vertices.append(copy.deepcopy(temp))
                    temp[j] += 1
                    if check_vertex(temp):
                        if temp not in vertices:
                            vertices.append(copy.deepcopy(temp))
            if changed:
                vertices.pop(i)
                continue
            i += 1
        for vertex in vertices:
            result = dot_product(vertex,self.my_equation)
            if result < minimum:
                minimum = result
                ans = vertex
        return ans