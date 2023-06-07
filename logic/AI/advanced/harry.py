from logic import *

P = Symbol("Tuesday")
Q = Symbol("raining")
R = Symbol("Harry")

KB = KnowledgeBase(Implication(And(P,Not(Q)),R),P,Not(Q))
query = R

ans = checkModel(KB,query)

print(ans)
print(KB.formula())