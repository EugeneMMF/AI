# this is to represent uncertainty in computers and hence solve uncertainty problems

# directed graph
# each node represents a random variable
# arrow from X to Y means X is a parent of Y
# each node X has a probability distribution P( X | Parents(X) )

# class to represent nodes.
class Node():
    def __init__(self,distribution,name):
        self.name = name
        self.distribution = distribution
        self.parents = []
    
    def __eq__(self, other):
        return isinstance(other,Node) and other.name == self.name

    def __hash__(self):
        return hash("node",hash(self.name))

# class for Discrete Probability Distribution
class DiscreteProbabilityDistribution():
    def __init__(self, values):
        if isinstance(values,dict):
            self.values = dict(values)
        else:
            raise Exception("must be dict")
    

    # returns the probability of a certain thing happening given by case
    def getValue(self, case):
        try:
            return self.values[case]
        except:
            raise Exception("not a member")

    # returns a set of all the possibilities of the random variable given by this distribution
    def possibilities(self):
        return set(self.values.keys())

# class for Conditional Probability Distribution
class ConditionalProbabilityDistribution():
    def __init__(self, values, *args):
        # values holds the conditional probability values
        # [ [{parent: value, parent2: value2, ...}, random value, probability], [...], ... ]
        self.values = values
        # dependencies are the nodes for the parents of the current node
        self.dependencies = list(args)
    
    # to get the value of the probability given a case
    def getValue(self, case):
        i = 0
        a = {}
        while i < len(self.values):
            if case in self.values[i]:
                value = 
            

    # returns a set of all the possibilities of the random variable given by the distribution
    def possibilities(self):
        poss = set()
        for value in self.values:
            poss.add(value[len(self.values)-2])
        return poss

# class for Bayesian Network
class BayesianNetwork():
    def __init__(self):
        self.nodes = []
    
    def addNodes(self, *args):
        for arg in args:
            self.nodes.append(arg)
    
    def addEdge(self, origin, terminal):
        if isinstance(terminal,Node) and isinstance(origin,Node):
            terminal.parents.append(origin)
        else:
            raise Exception("Invalid Operation. Must both be Node objects")
    
    def calculateProbability(self, conditions):
        def calculate(node, conditions):
            if isinstance(node,Node):
                if node.name in conditions.keys():
                    value = conditions[node.name]
                    if isinstance(node.distribution,DiscreteProbabilityDistribution):
                        return node.distribution[value]
                    elif isinstance(node.distribution,ConditionalProbabilityDistribution):
                        return calculate(node.parents)
