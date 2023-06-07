# imports
from cmath import isfinite
import copy
from dis import dis
from random import random
from sys import breakpointhook
import sys

# nodes represent random variables
# arrows have a probability distribution associated with them
# arrow from X to Y means X is a parent of Y

# class for the nodes
class Node():
    def __init__(self,distribution,name):
        self.distribution = distribution
        self.name = name
        self.parents = []
    
    # method to add parents to the node
    def addParent(self,parent):
        self.parents.append(parent)
    
    # method to get the possible values
    def get_values(self):
        return self.distribution.get_values()
    
# class for discrete probability table
class DiscreteProbabilityTable():
    def __init__(self,distribution):
        if isinstance(distribution,dict):
            self.distribution = distribution
        else:
            raise Exception("Discrete Distribution Table takes dict as its value")
    
    # method to get the possible values
    def get_values(self):
        return list(self.distribution.keys())
    
# class for conditional probability table
class ConditionalProbabilityTable():
    def __init__(self,distribution,*parents):
        self.distribution = []
        self.parents = []
        if len(parents) == 0:
            raise Exception("Conditional Probability Table must have parents")
        elif isinstance(distribution, list):
            for dist in distribution:
                if isinstance(dist,list) and len(dist) == len(parents)+2:
                    self.distribution.append(dist)
                else:
                    raise Exception("Conditional Probability Table takes a list of lists as its value")
        else:
            raise Exception("Conditional Distribution Table takes list as its value")
    
    def get_values(self):
        values = set()
        for arr in self.distribution:
            values.add(arr[-2])
        return list(values)

# class for Bayesian Network
class BayesianNetwork():
    def __init__(self):
        self.nodes = []
    
    # method to add nodes to the network
    def add_nodes(self, *states):
        for node in states:
            if isinstance(node, Node):
                self.nodes.append(node)
            else:
                raise Exception("Argument must be of class Node")
    
    # method to add arrows between the nodes in the network
    def add_edge(self,parent,child):
        if isinstance(parent,Node) and isinstance(child,Node):
            if child in self.nodes and parent in self.nodes:
                child.addParent(parent)
            else:
                raise Exception(f"First add the nodes to the network using {self.name}.add_node({child.name}) and {self.name}.add_node({parent.name})")
        else:
            raise Exception("Arguments must be of class Node")
    
    # method to generate samples
    def generate_samples(self,samples):
        # sub method to give the value of a random variable based on a distribution and a number from 0 to 1
        def sampling(value,distribution):
            i = 0
            keys = list(distribution.keys())
            a = distribution[keys[0]]
            while a < value:
                i += 1
                a += distribution[keys[i]]
            return keys[i]
        
        allSamples = []
        for i in range(1):
            nodes = copy.deepcopy(self.nodes)
            currentEvidence = {}
            order = []
            while len(nodes) != 0:
                i = 0
                while i < len(nodes):
                    if isinstance(nodes[i].distribution,DiscreteProbabilityTable):
                        currentEvidence.update({nodes[i].name:sampling(random(),nodes[i].distribution.distribution)})
                        nodes.pop(i)
                        order.append(i)
                        break
                    elif isinstance(nodes[i].distribution,ConditionalProbabilityTable):
                        requiredParents = set()
                        requiredArr = []
                        for parent in nodes[i].parents:
                            requiredParents.add(parent.name)
                            requiredArr.append(parent.name)
                        if len(requiredParents.difference(set(currentEvidence.keys()))) == 0:
                            passingEvidence = []
                            for key in requiredArr:
                                passingEvidence.append(currentEvidence[key])
                            distribution = {}
                            for dist in nodes[i].distribution.distribution:
                                if dist[:-2] == passingEvidence:
                                    distribution.update({dist[-2]:dist[-1]})
                            currentEvidence.update({nodes[i].name:sampling(random(),distribution)})
                            nodes.pop(i)
                            order.append(i)
                            break
                    i += 1
            allSamples.append(currentEvidence)
        for i in range(samples - 1):
            nodes = copy.deepcopy(self.nodes)
            currentEvidence = {}
            while len(nodes) != 0:
                for i in order:
                    if isinstance(nodes[i].distribution,DiscreteProbabilityTable):
                        currentEvidence.update({nodes[i].name:sampling(random(),nodes[i].distribution.distribution)})
                        nodes.pop(i)
                        continue
                    elif isinstance(nodes[i].distribution,ConditionalProbabilityTable):
                        requiredParents = set()
                        requiredArr = []
                        for parent in nodes[i].parents:
                            requiredParents.add(parent.name)
                            requiredArr.append(parent.name)
                        if len(requiredParents.difference(set(currentEvidence.keys()))) == 0:
                            passingEvidence = []
                            for key in requiredArr:
                                passingEvidence.append(currentEvidence[key])
                            distribution = {}
                            for dist in nodes[i].distribution.distribution:
                                if dist[:-2] == passingEvidence:
                                    distribution.update({dist[-2]:dist[-1]})
                            currentEvidence.update({nodes[i].name:sampling(random(),distribution)})
                            nodes.pop(i)
                            continue
                    i += 1
            allSamples.append(currentEvidence)
        return allSamples

    # method to predict probability using samples and given evidence
    def predict_probability_by_regection_sampling(self,allSamples,evidence):
        count = 0
        evidenceKeys = evidence.keys()
        evidenceCount = {}
        for node in self.nodes:
            evidenceCount[node.name] = {}
        for samp in allSamples:
            hold = True
            for evid in evidenceKeys:
                if evidence[evid] != samp[evid]:
                    hold = False
                    break
            if hold:
                count += 1
                for s in samp.keys():
                    if samp[s] in evidenceCount[s].keys():
                        evidenceCount[s][samp[s]] += 1
                    else:
                        evidenceCount[s][samp[s]] = 1
        for c in evidenceCount.keys():
            for d in evidenceCount[c].keys():
                evidenceCount[c][d] /= count
        return evidenceCount
    
    # method to give probability of some evidence occuring using samples
    def probability_by_regection_sampling(self,allSamples,evidence):
        samples = len(allSamples)
        count = 0
        evidenceKeys = evidence.keys()
        evidenceCount = {}
        for node in self.nodes:
            evidenceCount[node.name] = {}
        for samp in allSamples:
            hold = True
            for evid in evidenceKeys:
                if evidence[evid] != samp[evid]:
                    hold = False
                    break
            if hold:
                count += 1
                for s in samp.keys():
                    if samp[s] in evidenceCount[s].keys():
                        evidenceCount[s][samp[s]] += 1
                    else:
                        evidenceCount[s][samp[s]] = 1
        for c in evidenceCount.keys():
            for d in evidenceCount[c].keys():
                evidenceCount[c][d] /= count
        return count/samples
    
    # method to calculate the probability of evidence occuring given that all the random variables have been assigned a value
    def probability(self, evidence):
        if len(evidence) != len(self.nodes):
            raise Exception("Missing data in the evidence")
        nodes = copy.deepcopy(self.nodes)
        value = 1
        while len(nodes) != 0:
            i = 0
            while i < len(nodes):
                if isinstance(nodes[i].distribution,DiscreteProbabilityTable):
                    value *= nodes[i].distribution.distribution[evidence[nodes[i].name]]
                    nodes.pop(i)
                    break
                elif isinstance(nodes[i].distribution,ConditionalProbabilityTable):
                    requiredParents = set()
                    requiredArr = []
                    for parent in nodes[i].parents:
                        requiredParents.add(parent.name)
                        requiredArr.append(parent.name)
                    if len(requiredParents.difference(set(evidence.keys()))) == 0:
                        passingEvidence = []
                        for key in requiredArr:
                            passingEvidence.append(evidence[key])
                        for dist in nodes[i].distribution.distribution:
                            if dist[:-2] == passingEvidence and dist[-2] == evidence[nodes[i].name]:
                                value *= dist[-1]
                                break
                        nodes.pop(i)
                        break
                i += 1
        return value

    # method to predict the probability of some evidence
    def predict_probability(self, evidence):
        def partial_prob(nodes,evidence):
            if len(nodes) != 0:
                # get some node, get its values, lock for each and send recursively
                i = 0
                value = 0
                while i < len(nodes):
                    nodesCopy = copy.deepcopy(nodes)
                    evidenceCopy = copy.deepcopy(evidence)
                    node = nodesCopy.pop(i)
                    values = node.get_values()
                    evidenceCopy.update({node.name:values[0]})
                    for val in values:
                        evidenceCopy[node.name] = val
                        value += partial_prob(nodesCopy,evidenceCopy)
                    i += 1
            else:
                value = self.probability(evidence)
            return value
        
        nodesCopy = copy.deepcopy(self.nodes)
        evidenceCopy = copy.deepcopy(evidence)
        i = 0
        results = {}
        while i < len(nodesCopy):
            if nodesCopy[i].name in list(evidence.keys()):
                results[nodesCopy[i].name] = evidence[nodesCopy[i].name]
                nodesCopy.pop(i)
                continue
            else:
                results[nodesCopy[i].name] = {}
                values = nodesCopy[i].get_values()
                for val in values:
                    results[nodesCopy[i].name][val] = 0
                i += 1
        probabilityOfEvidence = partial_prob(nodesCopy,evidenceCopy)
        nodes = copy.deepcopy(nodesCopy)
        i = 0
        while i < len(nodes):
            nodesCopy = copy.deepcopy(nodes)
            evidenceCopy = copy.deepcopy(evidence)
            node = nodesCopy.pop(i)
            values = node.get_values()
            evidenceCopy.update({node.name:values[0]})
            total = 0
            for val in values:
                evidenceCopy[node.name] = val
                results[node.name][val] = partial_prob(nodesCopy,evidenceCopy) / probabilityOfEvidence
                total += results[node.name][val]
            # normalise
            for val in values:
                results[node.name][val] /= total
            i += 1
        return results

    # method to predict probability by likelihood weighting
    def predict_probability_by_likelihood_weighting(self,number,evidence):
        def partial_prob(nodes,evidence):
            if len(nodes) != 0:
                # get some node, get its values, lock for each and send recursively
                i = 0
                value = 0
                while i < len(nodes):
                    nodesCopy = copy.deepcopy(nodes)
                    evidenceCopy = copy.deepcopy(evidence)
                    node = nodesCopy.pop(i)
                    values = node.get_values()
                    evidenceCopy.update({node.name:values[0]})
                    for val in values:
                        evidenceCopy[node.name] = val
                        value += partial_prob(nodesCopy,evidenceCopy)
                    i += 1
            else:
                value = self.probability(evidence)
            return value
        
        def sampling(value,distribution):
            i = 0
            keys = list(distribution.keys())
            a = distribution[keys[0]]
            while a < value:
                i += 1
                a += distribution[keys[i]]
            return keys[i]

        def generate_samples_given_evidence(number,evidence):
            nds = copy.deepcopy(self.nodes)
            nodes = set()
            for nd in nds:
                nodes.add(nd.name)
            evidenceKeys = set(evidence.keys())
            # list of node names not in the evidence
            nodesMain = list(nodes.difference(evidenceKeys))
            nodes = copy.deepcopy(self.nodes)
            # remove all nodes that are in the evidence from the list of nodes
            i = 0
            while i < len(nodes):
                if not nodes[i].name in nodesMain:
                    nodes.pop(i)
                else:
                    i += 1
            evidenceCopy = copy.deepcopy(evidence)
            nodesMain = copy.deepcopy(nodes)
            allSamples = []
            order = []
            i = 0
            while i < len(nodes):
                if isinstance(nodes[i].distribution,DiscreteProbabilityTable):
                    evidenceCopy.update({nodes[i].name:sampling(random(),nodes[i].distribution.distribution)})
                    nodes.pop(i)
                    order.append(i)
                    i = -1
                elif isinstance(nodes[i].distribution,ConditionalProbabilityTable):
                    evidenceCopyKeys = set(evidenceCopy.keys())
                    nodeParents = set()
                    nodesP = []
                    for node in nodes[i].parents:
                        nodeParents.add(node.name)
                        nodesP.append(node.name)
                    if len(nodeParents.difference(evidenceCopyKeys)) == 0:
                        val = []
                        for n in nodesP:
                            val.append(evidenceCopy[n])
                        distribution = {}
                        for value in nodes[i].distribution.distribution:
                            if val == value[:-2]:
                                distribution.update({value[-2]:value[-1]})
                        evidenceCopy.update({nodes[i].name:sampling(random(),distribution)})
                        nodes.pop(i)
                        order.append(i)
                        i = -1
                i += 1
            allSamples.append(evidenceCopy)
            for t in range(number - 1):
                nodes = copy.deepcopy(nodesMain)
                evidenceCopy = copy.deepcopy(evidence)
                for i in order:
                    if isinstance(nodes[i].distribution,DiscreteProbabilityTable):
                        evidenceCopy.update({nodes[i].name:sampling(random(),nodes[i].distribution.distribution)})
                        nodes.pop(i)
                    elif isinstance(nodes[i].distribution,ConditionalProbabilityTable):
                        evidenceCopyKeys = set(evidenceCopy.keys())
                        nodeParents = set()
                        nodesP = []
                        for node in nodes[i].parents:
                            nodeParents.add(node.name)
                            nodesP.append(node.name)
                        if len(nodeParents.difference(evidenceCopyKeys)) == 0:
                            val = []
                            for n in nodesP:
                                val.append(evidenceCopy[n])
                            distribution = {}
                            for value in nodes[i].distribution.distribution:
                                if val == value[:-2]:
                                    distribution.update({value[-2]:value[-1]})
                            evidenceCopy.update({nodes[i].name:sampling(random(),distribution)})
                            nodes.pop(i)
                allSamples.append(evidenceCopy)
            return allSamples
        
        def get_likelihood(allSamples):
            likelihoods = []
            for sample in allSamples:
                p1 = partial_prob([],sample)
                likelihoods.append(1-p1)
            return likelihoods

        allSamples = generate_samples_given_evidence(number,evidence)
        nodes = copy.deepcopy(self.nodes)
        nodeKeys = set()
        for node in nodes:
            nodeKeys.add(node.name)
        nodeKeys = nodeKeys.difference(evidence.keys())
        nodesWithout = copy.deepcopy(nodes)
        i = 0
        while i < len(nodesWithout):
            if not nodesWithout[i].name in list(evidence.keys()):
                nodesWithout.pop(i)
            i += 1
        likelihoods = get_likelihood(allSamples)
        count = 0
        evidenceCount = {}
        for node in self.nodes:
            evidenceCount[node.name] = {}
        i = 0
        for samp in allSamples:
            count += likelihoods[i]
            for s in samp.keys():
                if samp[s] in evidenceCount[s].keys():
                    evidenceCount[s][samp[s]] += likelihoods[i]
                else:
                    evidenceCount[s][samp[s]] = likelihoods[i]
            i += 1
        for c in evidenceCount.keys():
            for d in evidenceCount[c].keys():
                evidenceCount[c][d] /= count
        return evidenceCount

# class for Markov model
class MarkovChain():
    def __init__(self,start,transition):
        self.start = start
        self.transition = transition

    def sample(self,number):
        def sampling(value,distribution):
            i = 0
            keys = list(distribution.keys())
            a = distribution[keys[0]]
            while a < value:
                i += 1
                a += distribution[keys[i]]
            return keys[i]
        
        states = []
        i = 0
        if isinstance(self.start,DiscreteProbabilityTable):
            j = 0
            val1 = []
            while j < len(self.transition.distribution[0]) - 2:
                val1.append(sampling(random(),self.start.distribution))
                j += 1
        else:
            val1 = self.start
        if isinstance(val1,list):
            for v in val1:
                states.append(v)
                i += 1
        else:
            states.append(val1)
        while i < number - 1:
            distribution = {}
            prev = len(self.transition.distribution[0])-2
            for dist in self.transition.distribution:
                if (dist[:-2]) == (states[len(states)-prev:]):
                    distribution.update({dist[-2]:dist[-1]})
            states.append(sampling(random(),distribution))
            i += 1
        return states

# class for hidden markov model
class HiddenMarkovModel():
    def __init__(self,transition,states,start,state_names):
        self.transition = transition
        self.states = states
        self.start = start
        self.state_names = state_names
    
    # method to predict the most likely states given some observations
    def most_likely(self,observations):
        predicted_states = []
        start = copy.deepcopy(self.start)
        # look at first observation and get the probability of each state given the observation
        j = 0
        while j < len(observations):
            observation = observations[j]
            i = 0
            distribution = {}
            count = 0
            while i < len(self.states):
                a = self.states[i].distribution[observation] * start[i]
                distribution[self.state_names[i]] = a
                count += a
                i += 1
            max = 0
            key = ""
            no = 0
            keyno = 0
            for dist in distribution.keys():
                distribution[dist] /= count
                if distribution[dist] > max:
                    max = distribution[dist]
                    key = dist
                    keyno = no
                no += 1
            predicted_states.append(key)
            # start = list(distribution.values())
            start = self.transition[keyno]
            print(j+1,start)
            j += 1
        return predicted_states