import logging

class Tree(object):

    def __init__(self, get_cost):
        self.get_cost = get_cost
        self.incumbent = None

    def explore(self):
        logging.shutdown()
        logging.basicConfig(
            format='%(message)s',
            filename='branch_and_bound.log',
            filemode='w',
            level=logging.INFO
            )
        child = Node(None, None, self.get_cost, self.get_upper_bound())
        while child is not None:
            parent = child
            child = parent.get_child(self.get_cost, self.get_upper_bound())
            while child is None:
                parent = parent.parent
                if parent is None:
                    self.exploration_completed()
                    return
                child = parent.get_child(self.get_cost, self.get_upper_bound())
            if child.integer:
                self.new_incumbent(child)
                child = child.parent

    def get_upper_bound(self):
        if self.incumbent is None:
            return np.inf
        else:
            return incumbent.cost

    def new_incumbent(self, node):
        self.incumbent = node
        logging.info('New upper bound:')
        logging.info('Cost -> ' + str(node.cost))
        logging.info('Path -> ' + str(node.path()))

    def exploration_completed(self):
        logging.info('Exploration completed:')
        if self.incumbent is None:
            logging.info('No feasible path found.')
        else:
            logging.info('Optimal cost -> ' + str(self.incumbent.cost))
            logging.info('Optimal path -> ' + str(self.incumbent.path()))

class Node(object):

    def __init__(self, parent, index, get_cost, upper_bound):
        self.parent = parent
        self.index = index
        self.cost, self.integer, self.childern_order, self.others = get_cost(self.path())
        self.feasible = self.cost is not None
        self.pruned = self.cost > upper_bound
        self.children = []
        self.node_report()

    def path(self):
        if self.parent is None:
            return []
        return self.parent.path() + [self.index]

    def get_child(self, get_cost, upper_bound):
        for index in self.childern_order[len(self.children):]:
            child = Node(self, index, get_cost, upper_bound)
            if child.feasible and not child.pruned:
                self.children.append(child)
                return child
            else:
                self.children.append(None)
        return None

    def node_report(self):
        logging.info('Explored path -> ' + str(self.path()))
        if not self.feasible:
            logging.info(' unfeasible')
        elif self.pruned:
            logging.info(' pruned, lower bound = ' + str(self.cost))
        else:
            if self.others['binaries'] is not None: ########### schifo qui
                logging.info(' deepened, binaries = ' + str([round(i, 4) for i in self.others['binaries']]))
            else:
                logging.info(' deepened')
