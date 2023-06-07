from uncert import *

rain = Node(DiscreteProbabilityTable({
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
],rain), name = "maintenance")

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
],rain,maintenance),name = "train")

appointment = Node(ConditionalProbabilityTable([
    ["on time","attend",0.9],
    ["on time","miss",0.1],
    ["delayed","attend",0.6],
    ["delayed","miss",0.4]
],train),name = "appointment")

model = BayesianNetwork()
model.add_nodes(rain,maintenance,train,appointment)

model.add_edge(rain,maintenance)
model.add_edge(rain, train)
model.add_edge(maintenance, train)
model.add_edge(train, appointment)

allSamples = model.generate_samples(1000)
samples = copy.deepcopy(allSamples)

# result = model.predict_probability_by_regection_sampling(samples,{
#     # "appointment":"attend",
#     "maintenance": "yes",
#     # "rain":"none",
#     # "train":"on time"
#     })

# printable = ""
# for i in result.keys():
#     printable += f"{i}\n"
#     for j in result[i].keys():
#         printable += f"\t{j}: {result[i][j]}\n"
# printable += "\b"
# print(printable)

a = "##################################################################"
# samples = copy.deepcopy(allSamples)

# print(model.probability_by_regection_sampling(samples,{
#     "appointment":"attend"
# }))

# print(
#     model.probability({
#         "rain":"none",
#         "maintenance": "no",
#         "train": "on time",
#         "appointment": "miss"
#     })
# )
print(a)

result = (model.predict_probability({
    # "appointment":"attend",
    "maintenance": "yes",
    # "rain":"none",
    # "train":"on time"
}))

printable = ""
for i in result.keys():
    printable += f"{i}\n"
    if isinstance(result[i],dict):
        for j in result[i].keys():
            printable += f"\t{j}: {result[i][j]}\n"
    else:
        printable += f"\t{result[i]}\n"
printable += "\b"
print(printable)

print(a)

# result = model.predict_probability_by_likelihood_weighting(10000,{
#     # "appointment":"attend",
#     # "maintenance": "yes",
#     "rain":"none",
#     # "train":"on time"
#     })

# printable = ""
# for i in result.keys():
#     printable += f"{i}\n"
#     for j in result[i].keys():
#         printable += f"\t{j}: {result[i][j]}\n"
# printable += "\b"
# print(printable)
