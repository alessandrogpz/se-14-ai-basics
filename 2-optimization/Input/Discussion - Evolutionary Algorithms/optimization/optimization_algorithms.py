

from optimization.optimization_problem import Optimization_Problem


class Optimization_Algorithms():
    """
        Optimization class with different optimization method implementations
    """

    @classmethod
    def exhaustive_search(cls):
        """
            calculate the best values for toast and wait duration by searching all values exhaustively.
        """
        best_utility = float('-inf')
        best_solution = None

        for toast_duration in range(1,101):
            for wait_duration in range(1,101):
                utility = Optimization_Problem.utility(
                                toast_duration = toast_duration,
                                wait_duration = wait_duration)
                
                if utility > best_utility:
                    best_utility = utility
                    best_solution = (toast_duration,wait_duration)
        return best_solution, best_utility
    


    @classmethod
    def hill_climbing(cls):
        """
            calculate the best values for toast and wait duration by implementing Hill Climbing
        """
        # TODO: implement me
        return None, float('-inf')

    @classmethod
    def gradient_ascent(cls):
        """
            Calculate the best values for toast_duration, wait_duration and power.
            This will require a mixture of hill climbing and Gradient Ascent
        """
        # TODO: implement me
        return None, float('-inf')
    
