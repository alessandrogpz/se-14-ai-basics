
import random

from optimization.optimization_problem import Optimization_Problem

class Genetic_Algorithm():
    """
        Optimization class with different optimization method implementations
    """

    POPULATION_SIZE = 1000
    GENERATIONS = 20
    MUTATION_RATE = 0.2
    SURVIVAL_PERCENTAGE = 0.1

    ####### HELPER FUNCTIONS #######

    @classmethod
    def assign_value_randomly(cls, dict, feature, dict1, dict2):
        """
            Assigns the value of 'feature' in a dictionary 'dict'.
            The value is randomly taken from 'dict1' or 'dict2' 
            
        """
        if random.choice([True, False]): 
            dict[feature] = dict1[feature]
        else:
            dict[feature] = dict2[feature]  

    
        
    @classmethod
    def correct_interval(cls, value, min_value, max_value):
        """
            helper function for staying within correct interval.
            Corrects a value to stay between min_value and max_value (inclusively)
        """
        return max(min_value,min(max_value,value))

    @classmethod
    def maybe_mutate(cls, value,min_value,max_value, delta_value):
        """
            possibly mutates a value by a step of delta_value.
            The mutation happens with MUTATION_RATE probability and will increase or decrease the value with equal probability.
            The result is guaranteed to stay within min_value and max_value (inclusive).
        """
        delta = 0
        if random.random() < cls.MUTATION_RATE:
            delta = random.choice([-delta_value, delta_value])

        return cls.correct_interval(value + delta, min_value, max_value)


    def print_population_statistics(population_sorted):
        """
            Prints the statistics of a population
            This function assumes that the population is already sorted according to the utility function.
        """
        print("Population: {individuums:",len(population_sorted),"min:",population_sorted[0]["utility"], "max: ", population_sorted[-1]["utility"],"}")



    ####### MAIN EVOLUTIONARY ALGORITHM FUNCTIONS #######

    @classmethod
    def get_random_population(cls):
        """
            Calculates POPULATION_SIZE random individuums.
            returns the population.
        """
        population = []
        for _ in range(cls.POPULATION_SIZE):
            toast_dur = random.randrange(1,101)
            wait_dur = random.randrange(1,101)
            power = random.random() * 2
            toaster = random.randrange(1,11)
            
            individuum = {
                "toast_duration":toast_dur,
                "wait_duration":wait_dur,
                "power": power,
                "toaster": toaster,
                "utility": Optimization_Problem.utility(toast_dur, wait_dur, power,toaster)
            }
            population.append(individuum)
        return population
    
    @classmethod
    def get_survivors(cls, population_sorted):
        """
            Calculates surviving individuums.
            This function assume the population is already sorted according to utility.
            Individuums with lower fitness are removed.
        """
        cutoff_point = int(cls.POPULATION_SIZE - cls.POPULATION_SIZE *cls.SURVIVAL_PERCENTAGE)
        return population_sorted[cutoff_point:len(population_sorted)]

    @classmethod
    def crossover(cls, parent1, parent2):
        """
            Creates a child from two parents via crossover.

            For each parameter a random value from the parents is picked.
        """

        child={}
        cls.assign_value_randomly(child,"toast_duration",parent1,parent2)
        cls.assign_value_randomly(child,"wait_duration",parent1,parent2)
        cls.assign_value_randomly(child,"power",parent1,parent2)
        cls.assign_value_randomly(child,"toaster",parent1,parent2)
        return child

    @classmethod
    def mutate(cls, individuum):
        """
            Takes one individuum and mutates one or more of its parameters.
            Mutation of each parameter happens with a chance of MUTATION_RATE
        """
        # mutate toast_duration
        feature = random.choice(["toast_duration","wait_duration","power", "toaster"])
        # mutate!

        if(feature == "toast_duration"):
            individuum["toast_duration"] = cls.maybe_mutate(individuum["toast_duration"],1,100,10)
        elif(feature == "wait_duration"):
            individuum["wait_duration"] = cls.maybe_mutate(individuum["wait_duration"],1,100,10)
        elif(feature == "toaster"):
            individuum["toaster"] = cls.maybe_mutate(individuum["toaster"],1,10,1)
        else:
            individuum["power"] = cls.maybe_mutate(individuum["power"],0,2,0.2)


        return individuum

    @classmethod
    def find_maximum(cls):
        """
            calculate the best values for toast duration, wait duration, power and toaster.
        """
        # 1) initialize population
        population = cls.get_random_population()

        # repeating 2, 3 and 4
        for _ in range(cls.GENERATIONS):
            # sort population so that fittest individuums are on higher indices
            population_sorted = sorted(population, key=lambda k: k["utility"])
            cls.print_population_statistics(population_sorted)

            
            # 2) determine survivors based on fitness
            survivors = cls.get_survivors(population_sorted)

            # 3) create children (crossover & mutate)
            children = []
            while len(children) + len(survivors) < cls.POPULATION_SIZE:
                child = cls.crossover(random.choice(survivors),random.choice(survivors))
                child = cls.mutate(child)

                child["utility"] = Optimization_Problem.utility(child["toast_duration"],child["wait_duration"],child["power"],child["toaster"])
                children.append(child)

            #4) updte population
            population = survivors + children

        return (population[-1]["toast_duration"], population[-1]["wait_duration"], population[-1]["power"], population[-1]["toaster"]), population[-1]["utility"]



