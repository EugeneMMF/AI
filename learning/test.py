from learning import *
import csv

# model = K_Nearestneighbour(n_neighbours = 3)

model = Perceptron(learning_rate = 0.3,iterations = 100)

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

# Separate data into training and testing groups
random.shuffle(data)
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]

length = len(evidence)

threshold = int(length * 0.5)

training_set = evidence[:threshold]
training_labels = labels[:threshold]

test_set = evidence[threshold:]
test_label = labels[threshold:]

model.train(training_set,training_labels)

lt = []
for label in test_label:
    if label == "Authentic":
        lt.append(0)
    else:
        lt.append(1)

correct = 0
incorrect = 0
predictions = model.predict(test_set)
for i in range(len(predictions)):
    if test_label[i] == predictions[i]:
        correct += 1
    else:
        incorrect += 1

print("correct:",correct)
print("incorrect:",incorrect)
print("accuracy:",correct/(correct+incorrect)*100)
print(model)