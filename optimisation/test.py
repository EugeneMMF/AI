from time import time
from csp import *

domain = ["Mon","Tue","Wed"]

A = Node("A",domain)
B = Node("B",domain)
C = Node("C",domain)
D = Node("D",domain)
E = Node("E",domain)
F = Node("F",domain)
G = Node("G",domain)

CONSTRAINTS = [
    (A,B),
    (A,C),
    (B,C),
    (B,D),
    (B,E),
    (C,E),
    (C,F),
    (D,E),
    (E,F),
    (E,G),
    (F,G)
]

nodes = [A,B,C,D,E,F,G]

problem = ConstraintProblem()

problem.add_node(A,B,C,D,E,F,G)

for x,y in CONSTRAINTS:
    problem.add_constraint(lambda x,y: x!=y ,(x,y))

start = time()
print(problem.solve(preferences = [[lambda x:x == 'Wed' ,(E)],[lambda x:x == 'Tue' ,(B)]]),time()-start)

problem.add_constraint(lambda x:x=="Tue",E)
print(problem.solve())

# for node in problem.nodes:
    # print(node.domain)