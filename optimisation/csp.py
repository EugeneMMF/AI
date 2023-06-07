# we shall have nodes representing the variables
# each node has a domain of values it can take
# each node has a list of other nodes which have constraints with it

# class for nodes
from cmath import inf
import copy


class Node():
    def __init__(self,name,domain):
        self.name = name
        self.value = None
        self.domain = copy.deepcopy(domain) # must be of type list
        self.linked = []

    # method to add a link between a this node and another
    def add_link(self,node):
        self.linked.append(node)
    
    def __hash__(self):
        return hash("node",hash(self.name))

    def __eq__(self,other):
        return isinstance(other,Node) and other.name == self.name # and set(self.domain) == set(other.domain) and set(self.linked) == set(other.linked)
    
# class for the constraint satisfaction problem
class ConstraintProblem():
    def __init__(self):
        self.nodes = []
        self.nodes_copy = []
        self.constraints = [] # stores function types+
        self.applications = [] # stores the variables to have the constraint as a tuple
    
    def add_node(self,*args):
        for arg in args:
            if not arg in self.nodes:
                self.nodes.append(arg)
                self.nodes_copy.append(copy.deepcopy(arg))
    
    def reset_domains(self):
        i = 0
        while i < len(self.nodes):
            node = self.nodes[i]
            node.domain = copy.deepcopy(self.nodes_copy[i].domain)
            i += 1

    def add_constraint(self,constraint,application):
        self.constraints.append(constraint)
        app = []
        if isinstance(application,Node):
            app.append(application)
            self.applications.append(app)
        else:
            self.applications.append(list(application))
        if isinstance(application,tuple) and len(application) == 2:
            for node in application:
                for node2 in application:
                    if node != node2:
                        node.add_link(node2)
    
    def solve(self,**kwargs):
        # to enforce unary node consistency if it succeeds the nodes domains are changed and returns True if it fails it resets the nodes domains and returns False
        def enforce_unary():
            i = 0
            while i < len(self.constraints):
                if len(self.applications[i]) != 1:
                    i += 1
                    continue
                node = self.applications[i][0]
                j = 0
                while j < len(node.domain):
                    val = node.domain[j]
                    if not self.constraints[i](val):
                        node.domain.pop(j)
                        continue
                    j += 1
                if len(node.domain) == 0:
                    self.reset_domains()
                    return False
                i += 1
            return True

        # to enforce binary node consistency if it succeeds the nodes domains are changed and returns True if it fails it resets the nodes domains and returns False
        def enforce_binary():
            """go through all the constraints find those with 2 nodes,
            add the index of the constraints found to a queue
            loop while queue is not empty
            pop an index, get the nodes and look through node1s domain
            checking if each has a value in node to that satisfies the constraint
            otherwise pop the value in its domain
            if the domain is changed get all the indexes of the constraints
            that have the node1 in them and add them to the queue
            do the same for node2"""
            def update_queue(node,queue):
                i = 0
                while i < len(self.applications):
                    if len(self.applications[i]) == 2 and (self.applications[i][0] == node or self.applications[i][1] == node) and (i not in queue):
                        queue.append(i)
                    i += 1
                return queue
            
            queue = []
            i = 0
            while i < len(self.applications):
                if len(self.applications[i]) == 2:
                    queue.append(i)
                i += 1
            while len(queue) != 0:
                i = queue.pop()
                node1 = self.applications[i][0]
                node2 = self.applications[i][1]
                j = 0
                while j < len(node1.domain):
                    val1 = node1.domain[j]
                    failed = True
                    for val2 in node2.domain:
                        if self.constraints[i](val1,val2):
                            failed = False
                            break
                    if failed:
                        node1.domain.pop(j)
                        queue = update_queue(node1,queue)
                    if len(node1.domain) == 0:
                        self.reset_domains()
                        return False
                    j += 1
                j = 0
                while j < len(node2.domain):
                    val2 = node2.domain[j]
                    failed = True
                    for val1 in node1.domain:
                        if self.constraints[i](val1,val2):
                            failed = False
                            break
                    if failed:
                        node2.domain.pop(j)
                        queue = update_queue(node2,queue)
                    if len(node2.domain) == 0:
                        self.reset_domains()
                        return False
                    j += 1
            return True 

        # to solve with preferences
        def solve_preferences(preferences):
            def add_preference(preference,application):
                self.constraints.append(preference)
                app = []
                if isinstance(application,Node):
                    app.append(application)
                    self.applications.append(app)
                else:
                    self.applications.append(list(application))
            
            constraints_size = len(self.constraints)
            for preference in preferences:
                add_preference(preference[0],preference[1])
            result = solve_plain({})
            while len(self.constraints) > constraints_size:
                self.constraints.pop(-1)
                self.applications.pop(-1)
            self.reset_domains()
            if result:
                return result
            return solve_plain({})
            
        # to solve without any preferences
        def solve_plain(values):
            # first set the domains of the values already set
            for value in values.keys():
                 for node in self.nodes:
                    if node.name == value:
                        node.domain = []
                        node.domain.append(values[value])
            if not(enforce_unary() and enforce_binary()):
                return False
            if len(values) == len(self.nodes):
                return values
            for node in self.nodes:
                if len(node.domain) == 1:
                    values[node.name] = node.domain[0]
            if len(values) == len(self.nodes):
                return values
            # select a node not in the values
            nodes = []
            for node in self.nodes:
                if not node.name in list(values.keys()):
                    nodes.append(node)
            high = 0
            high_nodes = []
            for node in nodes:
                if len(node.linked) >= high:
                    high = len(node.linked)
                    high_nodes.append(node)
            low = inf
            for node in high_nodes:
                if len(node.domain) < low:
                    low_node = node
                    low = len(node.domain)
            for value in low_node.domain:
                values[low_node.name] = value
                result = solve_plain(values)
                if result:
                    return result
            return False

        if 'preferences' in kwargs:
            preferences = kwargs['preferences']
            if not isinstance(preferences,list):
                raise Exception("must be list")
            result = solve_preferences(preferences)
            self.reset_domains()
            return result
        else:
            result = solve_plain({})
            self.reset_domains()
            return result