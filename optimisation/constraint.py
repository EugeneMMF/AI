# we shall have nodes representing the variables
# each node has a domain of values it can take
# each node has a list of other nodes which have constraints with it

# class for nodes
import copy


class Node():
    def __init__(self,name,domain):
        self.name = name
        self.value = None
        self.domain = copy.deepcopy(domain) # must be of type list
        self.linked = []

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
        self.constraints = [] # stores function types
        self.application = [] # stores the variables to have the constraint as a tuple
    
    def add_node(self,*args):
        for arg in args:
            if not arg in self.nodes:
                self.nodes.append(arg)
    
    def add_constraint(self,constraint,application):
        self.constraints.append(constraint)
        app = []
        if isinstance(application,Node):
            app.append(application)
            self.application.append(app)
        else:
            self.application.append(list(application))
        if isinstance(application,tuple) and len(application) == 2:
            for node in application:
                for node2 in application:
                    if node != node2:
                        node.add_link(node2)
    
    def solve(self,**kwargs):
        if kwargs.get("preferences") != None:
            result = self.solve_with_preferences(kwargs['preferences'])
            values = {}
            if not result:
                return result
            for node in result:
                values[node.name] = node.domain[0]
            return values

        else:
            result = self.solve_without_preferences([])
            values = {}
            if not result:
                return result
            for node in result:
                values[node.name] = node.domain[0]
            return values
    
    def solve_without_preferences(self,values):
        def enforce_node_consistency():
            i = 0
            while i < len(self.constraints):
                constraint = self.constraints[i]
                nodes = self.application[i]
                if len(nodes) == 1: # this means it is a unary constraint
                    j = 0
                    node = nodes[0]
                    while j < len(self.application[i][0].domain):
                        if not constraint(self.application[i][0].domain[j]):
                            print(self.application[i][0].name,self.application[i][0].domain[j])
                            self.application[i][0].domain.pop(j)
                            continue
                        j += 1
                    if len(self.application[i][0].domain) == 0:
                        return False
                i += 1
            return True
        
        def enforce_binary_constraint():
            def get_constraints(node):
                i = 0
                numbers = []
                while i < len(self.constraints):
                    if node in self.application[i]:
                        numbers.append(i)
                    i += 1
                return numbers
            
            queue = []
            i = 0
            while i < len(self.constraints):
                if len(self.application[i]) > 1:
                    queue.append(i)
                i += 1
            while len(queue) != 0:
                i = queue.pop(0)
                node1 = self.application[i][0]
                node2 = self.application[i][1]
                constraint = self.constraints[i]
                j = 0
                while j < len(node1.domain):
                    changed = True
                    for val in node2.domain:
                        if constraint(node1.domain[j],val):
                            changed = False
                            break
                    if changed:
                        self.application[i][0].domain.pop(j)
                        for num in get_constraints(node1):
                            queue.append(num)
                        continue
                    j += 1
                j = 0
                while j < len(node2.domain):
                    changed = True
                    for val in node1.domain:
                        if constraint(val,node2.domain[j]):
                            changed = False
                            break
                    if changed:
                        self.application[i][1].domain.pop(j)
                        for num in get_constraints(node2):
                            queue.append(num)
                        continue
                    j += 1
                if len(node1.domain) == 0 or len(node2.domain) == 0:
                    return False
            return True
        
        nodes = copy.deepcopy(self.nodes)
        if len(values) == len(nodes):
            if enforce_node_consistency() and enforce_binary_constraint():
                return values
            else:
                self.nodes = copy.deepcopy(nodes)
                return False
        if not (enforce_node_consistency() and enforce_binary_constraint()):
            self.nodes = copy.deepcopy(nodes)
            return False
        i = 0
        unassigned = []
        while i < len(self.nodes):
            if not self.nodes[i] in values:
                unassigned.append(self.nodes[i])
            i += 1
        maxim = 0
        best = []
        for nd in unassigned:
            if len(nd.linked) >= maxim:
                best.append(nd)
                maxim = len(nd.linked)
        maxim = 0
        for nd in best:
            if len(nd.domain) >= maxim:
                node = nd
                maxim = len(nd.domain)
        domain = copy.deepcopy(node.domain)
        vals = copy.deepcopy(values)
        vals.append(node)
        ran = len(node.domain)
        for i in range(ran):
            node.domain = [node.domain[i]] # set domain to one value
            result = self.solve_without_preferences(vals)
            if result:
                return result
            node.domain = copy.deepcopy(domain)
            self.nodes = copy.deepcopy(nodes)
        return False
    
    def solve_with_preferences(self,preferences):
        problem2 = copy.deepcopy(self)
        for preference in preferences:
            constraint = preference[0]
            application = preference[1]
            self.add_constraint(constraint,application)
        result = self.solve_without_preferences([])
        if not result:
            self = problem2
            print('Could not satisfy preferences')
            return self.solve_without_preferences([])
        else:
            return result