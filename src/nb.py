#%%
import numpy as np
import uuid
from enum import Enum
from typing import Optional
from scipy.special import expit
from scipy.stats import norm
#%%
#######class####
#######################################################################
# Classes
#######################################################################

class MatingStrategy(Enum):    
    ONE_TO_ONE = "one_to_one"
    ALL_VS_ALL = "all_vs_all"
    MATING_TYPES = "mating_types"

class MatingType(Enum):
    A = "A"
    ALPHA = "alpha"

class AlternativeFitnessMethod(Enum):
    """Enumeration of available fitness calculation methods."""
    SHERRINGTON_KIRKPATRICK = "sherrington_kirkpatrick"  # Original complex method
    SINGLE_POSITION = "single_position"  # Simple method based on a single position
    ADDITIVE = "additive"  # Simple additive method with no interactions

class DiploidOrganism:
    def __init__(self, parent1, parent2, fitness_model="dominant", mating_type=None):
        if len(parent1.genome) != len(parent2.genome):
            raise ValueError("Parent genomes must have the same length.")
        
        self.allele1 = parent1.genome.copy()
        self.allele2 = parent2.genome.copy()
        self.fitness_model = fitness_model
        self.environment = parent1.environment
        self.id = str(uuid.uuid4())
        self.parent1_id = parent1.id
        self.parent2_id = parent2.id
        self.mating_type = mating_type
        
        # *** Store individual parent fitness values ***
        self.parent1_fitness = parent1.fitness
        self.parent2_fitness = parent2.fitness
        
        # Compute average parent fitness (used elsewhere) but not for the heatmap
        self.avg_parent_fitness = (parent1.fitness + parent2.fitness) / 2
        
        self.fitness = self.calculate_fitness()

    def _get_effective_genome(self):
        """
        Calculate effective genome based on inheritance model.
        
        For codominant model:
        - If alleles are the same (1,1) or (-1,-1): use that value
        - If alleles are different (1,-1) or (-1,1): use 0.5 * (allele1 + allele2)
        - If environment prefers 1 and alleles are (1,1): returns 1
        """
         # Special handling for genome size 1
        if len(self.allele1) == 1:
            if self.fitness_model == "dominant":
                return np.array([-1]) if -1 in [self.allele1[0], self.allele2[0]] else np.array([1])
            elif self.fitness_model == "recessive":
                return np.array([-1]) if self.allele1[0] == -1 and self.allele2[0] == -1 else np.array([1])
            elif self.fitness_model == "codominant":
                if self.allele1[0] == self.allele2[0]:
                    return self.allele1.copy()
                else:
                    return np.array([0])  # Average of -1 and 1
    
        
        if self.fitness_model == "dominant":
            return np.where((self.allele1 == -1) | (self.allele2 == -1), -1, 1)
        elif self.fitness_model == "recessive":
            return np.where((self.allele1 == -1) & (self.allele2 == -1), -1, 1)
        elif self.fitness_model == "codominant":
            # Create a mask for mixed alleles (1,-1 or -1,1)
            mixed_alleles = self.allele1 != self.allele2
            
            # For mixed alleles, calculate the average (will give 0.5 * (1 + -1) = 0)
            # For same alleles, use either allele (they're the same)
            effective = np.where(
                mixed_alleles,
                0.5 * (self.allele1 + self.allele2),  # Mixed case: average of alleles
                self.allele1  # Same alleles case: use either allele
            )            
            # Special case: if environment prefers 1 and both alleles are 1
            both_positive = (self.allele1 == 1) & (self.allele2 == 1)
            effective = np.where(both_positive, 1, effective)
            
            return effective
        else:
            raise ValueError(f"Unknown fitness model: {self.fitness_model}")

    def calculate_fitness(self):
        """Calculate fitness using the effective genome and the environment's calculation method."""
        effective_genome = self._get_effective_genome()
        return self.environment.calculate_fitness(effective_genome)

class OrganismWithMatingType:
    def __init__(self, organism, mating_type):
        self.organism = organism
        self.mating_type = mating_type

class Environment:
    """
    Represents an environment with a fitness landscape for simulating evolutionary dynamics.
    Attributes:
        genome_size (int): The size of the genome.
        beta (float): A parameter controlling the ruggedness of the fitness landscape. Default is 0.5.
        rho (float): A parameter controlling the correlation between genome sites in the fitness landscape. Default is 0.25.
        seed (int or None): A seed for the random number generator to ensure reproducibility. Default is None.
        h (numpy.ndarray): The initialized fitness contributions of individual genome sites.
        J (numpy.ndarray): The initialized interaction matrix between genome sites.
    Methods:
        calculate_fitness(genome):
            Calculates the fitness of a given genome based on the fitness landscape.
    """
    def __init__(self, genome_size, beta=0.5, rho=0.25, seed=None, 
                 fitness_method=AlternativeFitnessMethod.SHERRINGTON_KIRKPATRICK):
        self.genome_size = genome_size
        self.beta = beta
        self.rho = rho
        self.seed = seed
        self.fitness_method = fitness_method
        
        # Use seeded RNG only for environment initialization
        env_rng = np.random.default_rng(seed)
        
        # Initialize fitness landscape based on the method
        if fitness_method == AlternativeFitnessMethod.SHERRINGTON_KIRKPATRICK:
            # Original method - initialize h and J
            self.h = init_h(self.genome_size, self.beta, random_state=env_rng)
            self.J = init_J(self.genome_size, self.beta, self.rho, random_state=env_rng)
            self.alternative_params = None
        else:
            # Alternative method - initialize appropriate parameters
            self.h = None
            self.J = None
            self.alternative_params = init_alternative_fitness(
                self.genome_size, method=fitness_method, random_state=env_rng)
        
    def calculate_fitness(self, genome):
        """Calculate fitness for a genome based on this environment's landscape."""
        if self.fitness_method == AlternativeFitnessMethod.SHERRINGTON_KIRKPATRICK:
            # Original complex fitness calculation
            energy = compute_fit_slow(genome, self.h, self.J, F_off=0.0)
            return energy
        else:
            # Alternative simplified fitness calculation
            return calculate_alternative_fitness(genome, self.alternative_params)
    
    def get_fitness_description(self):
        """Return a human-readable description of the fitness calculation method."""
        if self.fitness_method == AlternativeFitnessMethod.SHERRINGTON_KIRKPATRICK:
            return "Sherrington-Kirkpatrick model (complex interactions)"
        elif self.fitness_method == AlternativeFitnessMethod.SINGLE_POSITION:
            pos = self.alternative_params["position"]
            return f"Single position model (position {pos} determines fitness)"
        elif self.fitness_method == AlternativeFitnessMethod.ADDITIVE:
            return "Additive model (each position contributes independently)"
        return "Unknown fitness method" 

class Organism:
    """
    Represents an organism in a simulated environment with a genome, fitness, and mutation capabilities.

    Attributes:
        id (str): A unique identifier for the organism.
        environment (Environment): The environment in which the organism exists.
        genome (numpy.ndarray): The genome of the organism, represented as an array of -1 and 1.
        generation (int): The generation number of the organism.
        parent_id (str or None): The ID of the parent organism, if applicable.
        mutation_rate (float): The probability of mutation at each genome site.
        fitness (float): The fitness value of the organism, calculated based on its genome and environment.

    Methods:
        calculate_fitness():
            Calculates and returns the fitness of the organism based on its genome and environment.

        mutate():
            Introduces mutations to the organism's genome based on the mutation rate and updates its fitness.
    """
    def __init__(self, environment, genome=None, generation=0, parent_id=None, 
                mutation_rate=None, genome_seed=None, mutation_seed=None):
        self.id = str(uuid.uuid4())
        self.environment = environment
        
        # Create a new RNG with the provided seed for genome initialization
        genome_rng = np.random.default_rng(genome_seed)
        
        # For mutation RNG, combine the base mutation seed with a unique value for this organism
        # This ensures each organism has its own RNG stream but remains reproducible
        if mutation_seed is not None:
            # Create a value unique to this organism based on generation and a random component
            # Use genome_rng to generate this component for reproducibility
            unique_addition = generation * 1000 + genome_rng.integers(1000)
            organism_mutation_seed = mutation_seed + unique_addition
        else:
            organism_mutation_seed = None
            
        # Use seeded RNG for mutations
        self.rng = np.random.default_rng(organism_mutation_seed)
        
        if genome is None:
            self.genome = genome_rng.choice([-1, 1], environment.genome_size)
        else:
            self.genome = genome.copy()
        self.generation = generation
        self.parent_id = parent_id
        self.mutation_rate = mutation_rate if mutation_rate is not None else 1.0/environment.genome_size
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        return self.environment.calculate_fitness(self.genome)

    def mutate(self):
        # Use the organism's own unseeded RNG for mutations
        mutation_sites = self.rng.random(len(self.genome)) < self.mutation_rate
        self.genome[mutation_sites] *= -1
        self.fitness = self.calculate_fitness()

    def reproduce(self, mutation_seed=None):
        # When creating children, we pass the base mutation seed
        # Each child will generate its unique seed in its __init__ method
        child1 = Organism(self.environment, genome=self.genome,
                        generation=self.generation + 1, parent_id=self.id,
                        mutation_rate=self.mutation_rate, mutation_seed=mutation_seed)
        
        child2 = Organism(self.environment, genome=self.genome,
                        generation=self.generation + 1, parent_id=self.id,
                        mutation_rate=self.mutation_rate, mutation_seed=mutation_seed)
        
        return child1, child2



#%%
def compute_fit_slow(sigma, his, Jijs, F_off=0.0):
    """
    Compute the fitness of the genome configuration sigma using full slow computation.

    Parameters:
    sigma (np.ndarray): The genome configuration (vector of -1 or 1).
    his (np.ndarray): The vector of site-specific contributions to fitness.
    Jijs (np.ndarray): The interaction matrix between genome sites.
    F_off (float): The fitness offset, defaults to 0.

    Returns:
    float: The fitness value for the configuration sigma.
    Divide by 2 because every term appears twice in symmetric case.
    """
    return sigma @ (his + 0.5 * Jijs @ sigma) - F_off

def init_J(N, beta, rho, random_state=None):
    """
    Initialize the coupling matrix for the Sherrington-Kirkpatrick model with sparsity.
    """
    if not (0 < rho <= 1):
        raise ValueError("rho must be between 0 (exclusive) and 1 (inclusive).")
    
    rng = np.random.default_rng(random_state)
    
    # Handle special case when N=1
    if N == 1:
        return np.zeros((1, 1))  # Return a 1x1 zero matrix
        
    sig_J = np.sqrt(beta / (N * rho))  # Adjusted standard deviation for sparsity
    
    # Initialize an empty upper triangular matrix (excluding diagonal)
    J_upper = np.zeros((N, N))
    
    # Total number of upper triangular elements excluding diagonal
    total_elements = N * (N - 1) // 2
    
    # Number of non-zero elements based on rho
    num_nonzero = int(np.floor(rho * total_elements))
    if num_nonzero == 0 and rho > 0:
        num_nonzero = 1  # Ensure at least one non-zero element if rho > 0
    
    # Get the indices for the upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(N, k=1)
    
    # Randomly select indices to assign non-zero Gaussian values
    if total_elements > 0 and num_nonzero > 0:
        selected_indices = rng.choice(total_elements, size=num_nonzero, replace=False)
        # Map the selected flat indices to row and column indices
        rows = triu_indices[0][selected_indices]
        cols = triu_indices[1][selected_indices]
        # Assign Gaussian-distributed values to the selected positions
        J_upper[rows, cols] = rng.normal(loc=0.0, scale=sig_J, size=num_nonzero)
    
    # Symmetrize the matrix to make Jij symmetric
    Jij = J_upper + J_upper.T

    return Jij

def init_h(N, beta, random_state=None):
    """
    Initialize the external fields for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    N : int
        The number of spins.
    beta : float
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        The external fields.
    """
    rng = np.random.default_rng(random_state)
    sig_h = np.sqrt(1 - beta)
    return rng.normal(0.0, sig_h, N)


#%%
def main():
    # Simulation parameters
    genome_size = 10
    population_size = 10
    generations = 3
    beta = 0.5
    rho = 0.25
    fitness_model = "dominant"  # "dominant", "recessive", "codominant"
    mutation_seed = 42

    # Initialize environment
    env = Environment(
        genome_size=genome_size,
        beta=beta,
        rho=rho,
        seed=123,
        fitness_method=AlternativeFitnessMethod.SHERRINGTON_KIRKPATRICK
    )

    print("Fitness method:", env.get_fitness_description())

    # Create initial population
    population = [
        Organism(env, generation=0, genome_seed=i, mutation_seed=mutation_seed)
        for i in range(population_size)
    ]

    print(f"Initial population (size {len(population)}):")
    for i, org in enumerate(population):
        print(f"Organism {i}: Fitness = {org.fitness:.4f}, Genome = {org.genome}")

    # Reproduce for specified generations
    for gen in range(1, generations + 1):
        print(f"\nGeneration {gen}:")
        next_gen = []

        # Simple one-to-one mating strategy
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[(i + 1) % len(population)]  # wrap around for even pairing

            # Create diploid organism
            diploid = DiploidOrganism(parent1, parent2, fitness_model=fitness_model)

            print(f"Diploid from {i} and {(i + 1) % len(population)} -> Fitness: {diploid.fitness:.4f}, Effective Genome: {diploid._get_effective_genome()}")

            # Reproduce to form haploid children from each diploid parent
            child1, child2 = parent1.reproduce(mutation_seed=mutation_seed)
            next_gen.extend([child1, child2])

        # Limit population to original size
        population = next_gen[:population_size]

#%%
if __name__ == "__main__":
    main()


# %%
