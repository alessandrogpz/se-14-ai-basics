import random
import math


class Frontier:
    """
    Data structure that stores a frontier of nodes.
    Provides functionality to select and remove nodes according to a search algorithm.
    """
    def __init__(self, goal, search_algorithm):
        self.frontier = []
        self.goal = goal
        self.search_algorithm = search_algorithm

    def add(self, node):
        """
            Add a node to the frontier. 
            New nodes are added to the back.
        """
        self.frontier.append(node)

    def contains_state(self, state):
        """
            Checks whether a node is already in the frontier.
        """
        return any(node.state == state for node in self.frontier)

    def empty(self):
        """
            Checks whether the frontier is empty.
        """
        return len(self.frontier) == 0

    def pop(self):
        """
            Removes a node from the frontier and returns it.
            The returned node is selected by function select_node.
        """
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.select_node()
            self.frontier.remove(node)
            return node


    def select_node(self):
        """
            Selects the next node to extend
        """
        # random search
        if self.search_algorithm == "RS":
            # select a random node from the frontier
            return random.choice(self.frontier)

        # depth-first search
        if self.search_algorithm == "DFS":
            return self.frontier[-1]
        
        # breadth-first search
        if self.search_algorithm == "BFS":
            return self.frontier[0]
        
        # heuristic search
        # calulate the euclidian distance between all frontier nodes compared to the goal
        # chooses the smallest possible option
        # euclidian distance formula: sqrt( (node.x - goal.x)^2 + (node.y - goal.y)^2 ) 
        if self.search_algorithm == "HS":
            goal_x = self.goal[0]
            goal_y = self.goal[1]

            def distance_to_goal(node):
                node_x = node.state[0]
                node_y = node.state[1]
                return math.sqrt((node_x - goal_x) ** 2 + (node_y - goal_y) ** 2)

            closest = distance_to_goal(self.frontier[0])
            choice = self.frontier[0]

            for f_node in range(len(self.frontier)):
                if distance_to_goal(self.frontier[f_node]) < closest:
                    closest = distance_to_goal(self.frontier[f_node])
                    choice = self.frontier[f_node]

            return choice
                
            
