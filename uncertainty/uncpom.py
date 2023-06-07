from pomegranate import *

class Node():
    def __init__(self,distribution,name):
        self.distribution = distribution
        self.name = name


rain = Node(DiscreteDistribution({
    "none": 0.7,
    "light": 0.2,
    "heavy": 0.1
}),name = "rain")

maintenance = Node(ConditionalProbabilityTable([
    ["none","yes",0.4],
    ["none","no",0.6],
    ["light","yes",0.2],
    ["light","no",0.8],
    ["heavy","yes",0.1],
    ["heavy","no",0.9],
],[rain.distribution]), name = "maintenance")

train = Node(ConditionalProbabilityTable([
    ["none","yes","on time",0.8],
    ["none","yes","delayed",0.2],
    ["none","no","on time",0.9],
    ["none","no","delayed",0.1],
    ["light","yes","on time",0.6],
    ["light","yes","delayed",0.4],
    ["light","no","delayed",0.7],
    ["light","no","on time",0.3],
    ["heavy","yes","delayed",0.4],
    ["heavy","yes","on time",0.6],
    ["heavy","no","delayed",0.5],
    ["heavy","no","on time",0.5]
],[rain.distribution,maintenance.distribution]),name = "train")

appointment = Node(ConditionalProbabilityTable([
    ["on time","attend",0.9],
    ["on time","miss",0.1],
    ["delayed","attend",0.6],
    ["delayed","miss",0.4]
],[train.distribution]),name = "appointment")

model = BayesianNetwork()
model.add_states(rain,maintenance,train,appointment)

model.add_edge(rain,maintenance)
model.add_edge(rain, train)
model.add_edge(maintenance, train)
model.add_edge(train, appointment)

model.bake()

print(model.predict_proba({
    "maintenance":"yes"
}))
# print(model.probability([["none","no","on time","attend"]]))