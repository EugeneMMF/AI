from time import time
from pomegranate import *
import numpy

st = time()
sun = DiscreteDistribution({
    "umbrella":0.3,
    "no umbrella":0.7
})

rain = DiscreteDistribution({
    "umbrella":0.9,
    "no umbrella":0.1
})

states = [sun,rain]

transition = numpy.array([[0.8,0.2],[0.3,0.7]])

start = numpy.array([0.5,0.5])

model = HiddenMarkovModel.from_matrix(
    transition,states,start,state_names = ["sun","rain"]
)
model.bake()

observations =[
    "umbrella",
    "umbrella",
    "no umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "no umbrella",
    "no umbrella"
]
# observations = [
#     "no umbrella",
#     "no umbrella",
#     "umbrella",
#     "umbrella",
#     "umbrella",
#     "umbrella",
#     "no umbrella",
#     "umbrella",
#     "umbrella"
# ]

print(model.predict(observations),time()-st)