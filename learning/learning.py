import copy
import random

# K_NearestNeighbour
# Perceptron uses perceptron rule to learn, updates weights based on w_i = w_i + &(predicted - output)*x_i
# SVM gives the maximum seperating boundary using support vectors and can think in higher order

class K_Nearestneighbour():
    def __init__(self,**kwargs):
        self.nodes = []
        self.outputs = []
        self.classes = []
        if 'n_neighbours' in kwargs:
            self.k = kwargs['n_neighbours']
        else:
            self.k = 1
    
    def train(self,all_data,all_output):
        for i in range(len(all_data)):
            data = all_data[i]
            output = all_output[i]
            self.nodes.append(data)
            self.outputs.append(output)
            if not output in self.classes:
                self.classes.append(output)
    
    def predict(self,all_data):
        def sqr(val):
            return val * val

        def merge_sort(arr,**kwargs):
            if kwargs.get('indices'):
                indices = kwargs['indices']
            else:
                indices = []
                for i in range(len(arr)):
                    indices.append(i)
            length = len(arr)
            if length == 1:
                return [arr,indices]
            mid = int(length/2)
            left_arr = arr[:mid]
            left_indices = indices[:mid]
            right_arr = arr[mid:]
            right_indices = indices[mid:]
            left = merge_sort(left_arr,indices = left_indices)
            left_arr = left[0]
            left_indices = left[1]
            right = merge_sort(right_arr,indices = right_indices)
            right_arr = right[0]
            right_indices = right[1]
            new_arr = []
            new_indices = []
            left_counter = 0
            right_counter = 0
            len_right = len(right[0])
            len_left = len(left[0])
            while len(new_arr) != length:
                if right_counter < len_right:
                    if left_counter < len_left:
                        if right_arr[right_counter] < left_arr[left_counter]:
                            new_arr.append(right_arr[right_counter])
                            new_indices.append(right_indices[right_counter])
                            right_counter += 1
                        else:
                            new_arr.append(left_arr[left_counter])
                            new_indices.append(left_indices[left_counter])
                            left_counter += 1
                    else:
                        new_arr.append(right_arr[right_counter])
                        new_indices.append(right_indices[right_counter])
                        right_counter += 1
                else:
                    new_arr.append(left_arr[left_counter])
                    new_indices.append(left_indices[left_counter])
                    left_counter += 1
            return [new_arr,new_indices]
        
        predictions = []
        k = self.k
        for data in all_data:
            distances = []
            for node in self.nodes:
                dist = 0
                for i in range(len(node)):
                    dist += sqr(data[i] - node[i])
                distances.append(dist)
            sorted_indices = merge_sort(distances)[1]
            result = {}
            for cl in self.classes:
                result[cl] = 0
            maximum = 0
            iteration = 0
            for i in sorted_indices:
                result[self.outputs[i]] += 1
                if result[self.outputs[i]] > maximum:
                    maximum = result[self.outputs[i]]
                    max_class = self.outputs[i]
                iteration += 1
                if iteration >= k:
                    break
            predictions.append(max_class)
        return predictions

class Perceptron():
    def __init__(self,**kwargs):
        self.weights = []
        self.classes = []
        if 'learnig_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
        else:
            self.learning_rate = 0.5
        if 'iterations' in kwargs and kwargs['iterations'] > 0:
            self.iterations = kwargs['iterations']
        else:
            self.iterations = 1

    def __str__(self):
        return "".join(str(le) + "," for le in self.weights)

    def train(self,data,outputs):
        def sign(val):
            if val > 0:
                return 1
            return -1
        # print(len(data),len(outputs))
        for output in outputs:
            if output not in self.classes:
                self.classes.append(output)
        # print(self.classes)
        for d in range(len(data[0]) + 1):
            self.weights.append(random.random())
        for it in range(self.iterations):
            for i in range(len(data)):
                datum_send = copy.deepcopy(data[i])
                send = []
                send.append(datum_send)
                # print(send)
                prediction = self.classes.index(self.predict(send)[0])
                expected = self.classes.index(outputs[i])
                for j in range(len(self.weights)):
                    try:
                        self.weights[j] += self.learning_rate * (expected - prediction) * (data[i][j])
                    except:
                        self.weights[j] += self.learning_rate * (expected - prediction)
    
    def predict(self,data):
        def dot_product(matrixa,matrixb):
            ans = 0
            for i in range(len(matrixa)):
                ans += (matrixa[i] * matrixb[i])
            return ans

        predictions = []
        for datum in data:
            moded = copy.deepcopy(datum)
            moded.append(1)
            result = dot_product(moded,self.weights)
            if result >= 0:
                predictions.append(self.classes[1])
            else:
                predictions.append(self.classes[0])
        return predictions