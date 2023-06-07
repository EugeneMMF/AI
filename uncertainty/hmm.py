from time import time
from uncert import *
st = time()
sun = DiscreteProbabilityTable({
    "umbrella":0.3,
    "no umbrella":0.7
})

rain = DiscreteProbabilityTable({
    "umbrella":0.9,
    "no umbrella":0.1
})

states = [sun,rain]

transition = [[0.8,0.2],[0.3,0.7]]

start = [0.5,0.5]

model = HiddenMarkovModel(
    transition,states,start,state_names = ["sun", "rain"]
)

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

print(model.most_likely(observations),time()-st)