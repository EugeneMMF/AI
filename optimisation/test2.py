from linear import *

problem = LinearProgramming()

problem.add_constraint(
    [
        ([5,4],80),
        ([10,20],200)
    ]
)
problem.set_equation(
    [180,200]
)
problem.set_bounds(
    [0,0]
)
print(problem.maximize_int())
print(problem.maximize())