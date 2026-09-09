

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
        def _custom_utility(toast_duration, wait_duration):
            return Optimization_Problem.utility(toast_duration, wait_duration), toast_duration, wait_duration

        def _in_range(toast_duration, wait_duration):
            return 1 <= toast_duration <= 100 and 1 <= wait_duration <= 100

        # Initial state
        toast_d = 50
        wait_d = 50
        current = _custom_utility(toast_d, wait_d)

        while True:
            neighbours = [(toast_d + 1, wait_d),
                          (toast_d - 1, wait_d),
                          (toast_d, wait_d + 1),
                          (toast_d, wait_d - 1)]

            all_utilities = [current]

            # Append valid neighbours to all_utilities
            for next_toast, next_wait in neighbours:
                if _in_range(next_toast, next_wait):
                    all_utilities.append(_custom_utility(next_toast, next_wait))

            # Compare neighbours - the first one (best_utilities) is the MSD
            best_utility, best_toast, best_wait = max(all_utilities)

            # No better neighbours (found local best)
            if best_utility <= current[0]:
                return (toast_d, wait_d), current[0]

            # Found a better neighbour, restart search
            current = (best_utility, best_toast, best_wait)
            toast_d = best_toast
            wait_d = best_wait

    @classmethod
    def gradient_ascent(cls):
        """
            Calculate the best values for toast_duration, wait_duration and power.
            This will require a mixture of hill climbing and Gradient Ascent
        """
        # TODO: implement me
        return None, float('-inf')
    
