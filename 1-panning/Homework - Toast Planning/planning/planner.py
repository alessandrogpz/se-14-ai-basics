
import random
from  planning.planning_problem import Planning_Problem 

class Planner():
    """
        Planner class with different planning method implementations
    """

    @classmethod
    def random_search(cls, start_state):
        """
            Implementation of random search.

            returns (path, step_count) where path is a list of actions to the goal and step_count is the number of states extended
        """

        open_list = [(start_state,[])] # this is a list of state,path. The path of the start state is empty.
        start_state["path"]  = [] # small hack: we add paths to the state in order to remember the path we travelled.
        step_count = 0
        
        while len(open_list) > 0:
            state, path = random.choice(open_list)
            step_count+= 1

            for action in Planning_Problem.actions:
                next_state = Planning_Problem.state_transition(state,action)
                next_path = path + [action]

                if Planning_Problem.goal(next_state):
                    return next_path, step_count
                else:
                    open_list.append((next_state,next_path))

        return None, step_count
    

    @classmethod
    def breadth_first_search(cls, start_state):
        """
            Implementation of Breadth First Search

            returns (path, step_count) where path is a list of actions to the goal and step_count is the number of states extended
        """

        # A hash map that ignore states already seen / visited
        def state_key(state):
            return (state["toaster_has_power"], state["toaster_is_on"], state["bread_location"], state["bread_state"])

        open_list = [(start_state,[])]
        seen = {state_key(start_state)}
        step_count = 0

        while len(open_list) > 0:
            state, path = open_list.pop(0)                                          # remove top queue element (pop() returns the element popped)
            step_count += 1  
            if Planning_Problem.goal(state):
                        return path, step_count
            else:
                for action in Planning_Problem.actions:                                               
                    next_state = Planning_Problem.state_transition(state, action)   # create a new state
                    if state_key(next_state) not in seen:                           # check if same state already happned, if so skip it, 
                        seen.add(state_key(next_state))                             #                                      otherwise append to state_key
                        open_list.append((next_state, path + [action]  ))           # append new action to path & push it down the queue


        return None, step_count
    

    @classmethod
    def shortest_toast_time_search(cls, start_state):
        """
            Implementation of a search algorithm that also optimized for the time it takes to toast a slice of bread.
            This time is measured within the state in attribute ["time"]

            returns (path, step_count) where path is a list of actions to the goal and step_count is the number of states extended
        """

        # Dijkstra's Implementation

        # A hash map that ignore states already seen / visited
        def state_key(state):
            return (state["toaster_has_power"], state["toaster_is_on"], state["bread_location"], state["bread_state"])

        open_list = [(start_state, [])]
        closed = set()
        step_count = 0

        while len(open_list) > 0:
            best_time = float("inf")
            state_choice = 0

            # Scan for the best (shortest) cumulative time among frontier nodes.
            for index in range(len(open_list)):
                current_time = open_list[index][0]["time"]

                if current_time < best_time:
                    best_time = current_time
                    state_choice = index

            # Pop the node with the best cumulative time
            state, path = open_list.pop(state_choice)

            # A state can stay in the frontier several iterations, reached by different routes
            # Append only only unique states (to prevent 'plug' -> 'unplug' -> 'plug' -> 'unplug' -> ...)
            if state_key(state) in closed:
                continue
            closed.add(state_key(state))
            step_count += 1

            # Checks for goal completion
            if Planning_Problem.goal(state):
                return path, step_count

            # Iterate through actions and append unique routes
            for action in Planning_Problem.actions:
                next_state = Planning_Problem.state_transition(state, action)
                if state_key(next_state) not in closed:
                    open_list.append((next_state, path + [action]))

        return None, step_count