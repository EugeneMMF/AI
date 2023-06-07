from uncert import *

start = DiscreteProbabilityTable({
    "sun":0.5,
    "rain":0.5
})


transition = ConditionalProbabilityTable([
    ["sun","sun","sun",0.8],
    ["sun","sun","rain",0.2],
    ["sun","rain","sun",0.3],
    ["sun","rain","rain",0.7],
    ["rain","sun","sun",0.6],
    ["rain","sun","rain",0.4],
    ["rain","rain","sun",0.1],
    ["rain","rain","rain",0.9]
],start,start)

# model = MarkovChain(["sun","sun"],transition)
# print(model.sample(50))

transition = ConditionalProbabilityTable([
    ["sun","sun",0.8],
    ["sun","rain",0.2],
    ["rain","sun",0.3],
    ["rain","rain",0.7]
],start)

model = MarkovChain(start,transition)

print(model.sample(50))