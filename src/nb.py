import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import itertools
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class NKModel:
    """
    Implementation of the NK Model for creating tunable fitness landscapes.
    
    Attributes:
        N (int): Number of genes in the genome
        K (int): Number of other genes each gene interacts with (epistasis parameter)
        interaction_matrix (np.ndarray): Matrix defining which genes interact with each other
        contribution_tables (list): Fitness contribution tables for each gene
        
    Methods:
        generate_random_interaction_matrix(): Creates a random interaction matrix
        generate_contribution_tables(): Creates random fitness contribution tables
        calculate_fitness(genome): Calculates fitness for a given genome
        generate_fitness_landscape(): Generates the complete fitness landscape
        find_local_optima(): Identifies all local optima in the landscape
        get_neighbors(genome): Gets all one-bit mutation neighbors of a genome
        visualize_landscape(): Visualizes the fitness landscape
        analyze_landscape_ruggedness(): Analyzes the ruggedness by counting local optima
    """
        
    def __init__(self, N, K, seed=None):
            """
            Initialize the NK Model.
            
            Args:
                N (int): Number of genes in the genome
                K (int): Number of other genes each gene interacts with (0 <= K < N)
                seed (int, optional): Random seed for reproducibility
            """
            if K >= N:
                raise ValueError(f"K must be less than N. Got K={K}, N={N}")
            
            self.N = N
            self.K = K
            self.rng = np.random.RandomState(seed)
            
            # Generate interaction matrix and contribution tables
            self.interaction_matrix = self.generate_random_interaction_matrix()
            self.contribution_tables = self.generate_contribution_tables()
            
            # Cache for fitness values to avoid recalculation
            self.fitness_cache = {}

    def generate_random_interaction_matrix(self):
        """
        Generate a random interaction matrix where each gene interacts with K other genes.
        
        Returns:
            np.ndarray: An N x (K+1) matrix where each row i contains the indices of 
                        genes that affect gene i (including i itself)
        """
        interaction_matrix = np.zeros((self.N, self.K + 1), dtype=int)
        
        for i in range(self.N):
            # Each gene always interacts with itself
            interaction_matrix[i, 0] = i
            
            # Select K other genes randomly
            other_genes = [j for j in range(self.N) if j != i]
            selected = self.rng.choice(other_genes, self.K, replace=False)
            interaction_matrix[i, 1:] = selected
            
        return interaction_matrix
    
    def generate_contribution_tables(self):
        """
        Generate random fitness contribution tables for each gene.
        
        Returns:
            list: A list of 2^(K+1) fitness contributions for each gene
        """
        contribution_tables = []
        
        for i in range(self.N):
            # Create a table for each possible state of the gene and its K interacting genes
            # There are 2^(K+1) possible states
            table = self.rng.random(2**(self.K + 1))
            contribution_tables.append(table)
            
        return contribution_tables
    
    def calculate_fitness(self, genome):
        """
        Calculate the fitness of a genome based on the NK model.
        
        Args:
            genome (np.ndarray or list): Binary genome of length N (0s and 1s)
            
        Returns:
            float: The fitness value of the genome
        """
        # Convert genome to tuple for caching
        genome_tuple = tuple(genome)
        
        # Return cached value if available
        if genome_tuple in self.fitness_cache:
            return self.fitness_cache[genome_tuple]
        
        fitness_contributions = []
        
        for i in range(self.N):
            # Get the state of gene i and its interacting genes
            interacting_genes = self.interaction_matrix[i]
            state = [genome[j] for j in interacting_genes]
            
            # Convert state to an index for the contribution table
            # Example: state [1,0,1] becomes 1*2^0 + 0*2^1 + 1*2^2 = 5
            index = sum(state[j] * (2 ** j) for j in range(len(state)))
            
            # Get the fitness contribution for this state
            contribution = self.contribution_tables[i][index]
            fitness_contributions.append(contribution)
        
        # Calculate overall fitness as the average of all contributions
        fitness = sum(fitness_contributions) / self.N
        
        # Cache the result
        self.fitness_cache[genome_tuple] = fitness
        
        return fitness
    
    def generate_fitness_landscape(self):
        """
        Generate the complete fitness landscape for all possible genomes.
        
        Returns:
            dict: A dictionary mapping genome tuples to fitness values
        """
        landscape = {}
        
        # Generate all possible binary genomes of length N
        all_genomes = list(itertools.product([0, 1], repeat=self.N))
        
        # Calculate fitness for each genome
        for genome in tqdm(all_genomes, desc=f"Generating landscape for K={self.K}"):
            landscape[genome] = self.calculate_fitness(genome)
            
        return landscape
    
    def get_neighbors(self, genome):
        """
        Get all one-bit mutation neighbors of a genome.
        
        Args:
            genome (tuple): The genome as a tuple
            
        Returns:
            list: List of neighbor genomes (as tuples)
        """
        neighbors = []
        
        for i in range(self.N):
            # Create a new genome with the i-th bit flipped
            neighbor = list(genome)
            neighbor[i] = 1 - neighbor[i]  # Flip 0 to 1 or 1 to 0
            neighbors.append(tuple(neighbor))
            
        return neighbors
    
    def find_local_optima(self, landscape=None):
        """
        Find all local optima in the fitness landscape.
        
        Args:
            landscape (dict, optional): Pre-generated fitness landscape
            
        Returns:
            list: List of genome tuples that are local optima
        """
        if landscape is None:
            landscape = self.generate_fitness_landscape()
            
        local_optima = []
        
        for genome, fitness in landscape.items():
            # Get all neighbors
            neighbors = self.get_neighbors(genome)
            
            # Check if this genome has higher fitness than all neighbors
            is_local_optimum = True
            for neighbor in neighbors:
                if landscape[neighbor] > fitness:
                    is_local_optimum = False
                    break
                    
            if is_local_optimum:
                local_optima.append(genome)
                
        return local_optima
    
    def analyze_landscape_ruggedness(self, num_samples=None):
        """
        Analyze the ruggedness of the landscape by counting local optima.
        
        Args:
            num_samples (int, optional): Number of random genomes to sample for fitness
            
        Returns:
            dict: Dictionary containing metrics about landscape ruggedness
        """
        # Generate the complete landscape if it's small enough
        if self.N <= 16:  # 2^16 = 65,536 genomes
            landscape = self.generate_fitness_landscape()
            local_optima = self.find_local_optima(landscape)
            
            # Calculate average fitness
            avg_fitness = sum(landscape.values()) / len(landscape)
            
            # Calculate fitness correlation
            fitnesses = list(landscape.values())
            mean_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses)
            
            return {
                'num_local_optima': len(local_optima),
                'fraction_local_optima': len(local_optima) / len(landscape),
                'avg_fitness': avg_fitness,
                'fitness_std': std_fitness
            }
        else:
            # For large N, sample the landscape
            if num_samples is None:
                num_samples = min(10000, 2**self.N)
                
            # Sample random genomes
            samples = []
            for _ in range(num_samples):
                genome = tuple(self.rng.randint(0, 2, self.N))
                samples.append((genome, self.calculate_fitness(genome)))
                
            # Calculate average fitness
            avg_fitness = sum(f for _, f in samples) / len(samples)
                
            return {
                'avg_fitness': avg_fitness,
                'note': 'Full landscape analysis not performed due to large N'
            }
    
    def visualize_landscape(self, max_dims=2):
        """
        Visualize the fitness landscape for small N.
        
        Args:
            max_dims (int): Maximum number of dimensions to visualize (1 or 2)
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        if self.N > max_dims:
            print(f"WARNING: N={self.N} is too large to visualize directly. Showing projection.")
        
        if max_dims == 1:
            # 1D visualization (for N=1)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            genomes = [(0,), (1,)]
            fitnesses = [self.calculate_fitness(g) for g in genomes]
            
            ax.bar([0, 1], fitnesses)
            ax.set_xlabel('Genome')
            ax.set_ylabel('Fitness')
            ax.set_title(f'NK Model Fitness Landscape (N={self.N}, K={self.K})')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['0', '1'])
            
        elif max_dims == 2:
            if self.N >= 2:
                # 2D visualization (for N=2 or projection of higher dims)
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # For N=2, visualize the complete landscape
                if self.N == 2:
                    # Generate all 4 possible genomes
                    x = np.array([0, 0, 1, 1])
                    y = np.array([0, 1, 0, 1])
                    genomes = [(x[i], y[i]) for i in range(4)]
                    fitnesses = [self.calculate_fitness(g) for g in genomes]
                    
                    # 3D surface plot
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_trisurf(x, y, fitnesses, cmap=cm.viridis, linewidth=0.2)
                    ax.set_xlabel('Gene 1')
                    ax.set_ylabel('Gene 2')
                    ax.set_zlabel('Fitness')
                    ax.set_title(f'NK Model Fitness Landscape (N={self.N}, K={self.K})')
                    
                # For N>2, create a heatmap projection
                else:
                    # Generate all combinations of the first two genes
                    x_vals = np.array([0, 0, 1, 1])
                    y_vals = np.array([0, 1, 0, 1])
                    
                    # For each combination, generate multiple random configurations of other genes
                    # and calculate average fitness
                    z = np.zeros((2, 2))
                    samples_per_cell = 50
                    
                    for i, x in enumerate([0, 1]):
                        for j, y in enumerate([0, 1]):
                            avg_fitness = 0
                            for _ in range(samples_per_cell):
                                # Generate random values for the other N-2 genes
                                other_genes = self.rng.randint(0, 2, self.N - 2)
                                genome = np.array([x, y, *other_genes])
                                avg_fitness += self.calculate_fitness(genome)
                            z[i, j] = avg_fitness / samples_per_cell
                    
                    # Create heatmap
                    sns.heatmap(z, annot=True, fmt=".3f", cmap="viridis", 
                              xticklabels=[0, 1], yticklabels=[0, 1],
                              cbar_kws={'label': 'Average Fitness'})
                    ax.set_xlabel('Gene 2')
                    ax.set_ylabel('Gene 1')
                    ax.set_title(f'NK Model Average Fitness Projection (N={self.N}, K={self.K})')
            else:
                # Fallback to 1D for N=1
                return self.visualize_landscape(max_dims=1)
                
        plt.tight_layout()
        return fig

def analyze_NK_model_ruggedness(N, K_values, num_runs=5, seed=42):
    """
    Analyze how landscape ruggedness changes with different K values.
    
    Args:
        N (int): Number of genes
        K_values (list): List of K values to test
        num_runs (int): Number of runs for each K value
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with analysis results
    """
    results = {
        'K_values': K_values,
        'avg_local_optima': [],
        'avg_fitness': [],
        'local_optima_stderr': [],
        'fitness_stderr': []
    }
    
    for K in K_values:
        local_optima_counts = []
        avg_fitnesses = []
        
        for run in range(num_runs):
            # Create model with different seed for each run
            model = NKModel(N=N, K=K, seed=seed + run)
            
            # Analyze landscape
            analysis = model.analyze_landscape_ruggedness()
            
            if 'num_local_optima' in analysis:
                local_optima_counts.append(analysis['num_local_optima'])
            avg_fitnesses.append(analysis['avg_fitness'])
        
        # Calculate average metrics across runs
        if local_optima_counts:
            results['avg_local_optima'].append(np.mean(local_optima_counts))
            results['local_optima_stderr'].append(np.std(local_optima_counts) / np.sqrt(len(local_optima_counts)))
        else:
            results['avg_local_optima'].append(None)
            results['local_optima_stderr'].append(None)
            
        results['avg_fitness'].append(np.mean(avg_fitnesses))
        results['fitness_stderr'].append(np.std(avg_fitnesses) / np.sqrt(len(avg_fitnesses)))
    
    return results

def plot_ruggedness_analysis(results):
    """
    Plot the results of the ruggedness analysis.
    
    Args:
        results (dict): Results from analyze_NK_model_ruggedness
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot number of local optima vs K
    if results['avg_local_optima'][0] is not None:
        ax1.errorbar(results['K_values'], results['avg_local_optima'], 
                    yerr=results['local_optima_stderr'], fmt='-o')
        ax1.set_xlabel('K (Epistatic Interactions)')
        ax1.set_ylabel('Average Number of Local Optima')
        ax1.set_title('Landscape Ruggedness vs K')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Local optima analysis not available for large N', 
                ha='center', va='center')
        ax1.set_title('Landscape Ruggedness (N too large)')
    
    # Plot average fitness vs K
    ax2.errorbar(results['K_values'], results['avg_fitness'], 
                yerr=results['fitness_stderr'], fmt='-o', color='orange')
    ax2.set_xlabel('K (Epistatic Interactions)')
    ax2.set_ylabel('Average Fitness')
    ax2.set_title('Average Fitness vs K')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_multiple_landscapes(N, K_values, seed=42):
    """
    Create visualization of landscapes with different K values.
    
    Args:
        N (int): Number of genes
        K_values (list): List of K values to visualize
        seed (int): Random seed for reproducibility
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    num_plots = len(K_values)
    fig = plt.figure(figsize=(5*num_plots, 5))
    
    for i, K in enumerate(K_values):
        model = NKModel(N=N, K=K, seed=seed)
        
        if N <= 2:
            # For N=1 or N=2, we can visualize exact landscape
            ax = fig.add_subplot(1, num_plots, i+1, projection='3d' if N==2 else None)
            
            if N == 1:
                genomes = [(0,), (1,)]
                fitnesses = [model.calculate_fitness(g) for g in genomes]
                ax.bar([0, 1], fitnesses)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['0', '1'])
                ax.set_xlabel('Gene Value')
                ax.set_ylabel('Fitness')
            else:  # N == 2
                x = np.array([0, 0, 1, 1])
                y = np.array([0, 1, 0, 1])
                genomes = [(x[i], y[i]) for i in range(4)]
                fitnesses = [model.calculate_fitness(g) for g in genomes]
                
                ax.plot_trisurf(x, y, fitnesses, cmap=cm.viridis, linewidth=0.2)
                ax.set_xlabel('Gene 1')
                ax.set_ylabel('Gene 2')
                ax.set_zlabel('Fitness')
                
            ax.set_title(f'K={K}')
        else:
            # For N>2, show heatmap projection
            ax = fig.add_subplot(1, num_plots, i+1)
            
            # Generate all combinations of the first two genes
            z = np.zeros((2, 2))
            samples_per_cell = 50
            
            for i1, x in enumerate([0, 1]):
                for j1, y in enumerate([0, 1]):
                    avg_fitness = 0
                    for _ in range(samples_per_cell):
                        # Generate random values for the other N-2 genes
                        other_genes = np.random.randint(0, 2, N - 2)
                        genome = np.array([x, y, *other_genes])
                        avg_fitness += model.calculate_fitness(genome)
                    z[i1, j1] = avg_fitness / samples_per_cell
            
            # Create heatmap
            sns.heatmap(z, annot=True, fmt=".3f", cmap="viridis", 
                      xticklabels=[0, 1], yticklabels=[0, 1],
                      cbar_kws={'label': 'Avg Fitness'}, ax=ax)
            ax.set_xlabel('Gene 2')
            ax.set_ylabel('Gene 1')
            ax.set_title(f'K={K} (Avg Fitness Projection)')
    
    plt.tight_layout()
    return fig

def create_3d_fitness_landscape(N, K_values, num_genes_to_plot=3, seed=42):
    """
    Create interactive 3D visualizations of fitness landscapes for different K values using Plotly.
    
    Args:
        N (int): Total number of genes
        K_values (list): List of K values to visualize
        num_genes_to_plot (int): Number of genes to include in the 3D plot (2 or 3)
        seed (int): Random seed for reproducibility
        
    Returns:
        plotly.graph_objects.Figure: Interactive 3D plot figure
    """
    if num_genes_to_plot not in [2, 3]:
        raise ValueError("num_genes_to_plot must be 2 or 3")
    
    # Filter K values to ensure they are less than N
    valid_K_values = [k for k in K_values if k < N]
    if not valid_K_values:
        valid_K_values = [0, min(1, N-1)]
    
    num_plots = len(valid_K_values)
    
    # Dynamically adjust the number of rows and columns based on the number of plots
    num_cols = min(3, num_plots)  # Maximum 3 columns per row
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate required rows
    print(f"Creating {num_rows} rows and {num_cols} columns for {num_plots} plots.")
    
    # Create specs for each subplot
    specs = [[{'type': 'scene'} for _ in range(num_cols)] for _ in range(num_rows)]
    
    # Create subplot titles
    subplot_titles = [f'K={K}' for K in valid_K_values]
    
    # Create subplot figure
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        specs=specs,
        subplot_titles=subplot_titles
    )
    
    for plot_idx, K in enumerate(valid_K_values):
        # Create model with specified K value
        model = NKModel(N=N, K=K, seed=seed)
        
        if num_genes_to_plot == 2:
            # For 2 genes with fitness as 3rd dimension
            # Generate all possible states for these 2 genes
            all_states = list(itertools.product([0, 1], repeat=2))
            x = [state[0] for state in all_states]
            y = [state[1] for state in all_states]
            
            # Calculate average fitness for each state
            z = []
            for g1, g2 in all_states:
                # For each state, generate and average multiple random configurations of other genes
                avg_fitness = 0
                num_samples = 50
                for _ in range(num_samples):
                    # Generate random values for other N-2 genes
                    other_genes = np.random.RandomState(seed).randint(0, 2, N - 2)
                    genome = np.array([g1, g2, *other_genes])
                    avg_fitness += model.calculate_fitness(genome)
                z.append(avg_fitness / num_samples)
            
            # Create scatter3d trace
            scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=10,
                    color=z,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Fitness")
                ),
                text=[f'Genome: [{g1}, {g2}]<br>Fitness: {fitness:.4f}' 
                      for (g1, g2), fitness in zip(all_states, z)],
                hoverinfo='text',
                name=f'K={K}'
            )
            
            # Create surface connecting the points
            surface_x, surface_y = np.meshgrid([0, 1], [0, 1])
            surface_z = np.array(z).reshape(2, 2)
            
            surface = go.Surface(
                x=surface_x,
                y=surface_y,
                z=surface_z,
                colorscale='Viridis',
                opacity=0.5,
                showscale=False,
            )
            
            # Calculate row and column for this subplot
            row_idx = plot_idx // num_cols + 1  # 1-indexed
            col_idx = plot_idx % num_cols + 1   # 1-indexed
            
            # Add traces to the figure
            fig.add_trace(scatter, row=row_idx, col=col_idx)
            fig.add_trace(surface, row=row_idx, col=col_idx)
            
            # Update axis labels
            fig.update_scenes(
                xaxis_title="Gene 1",
                yaxis_title="Gene 2",
                zaxis_title="Fitness",
                xaxis=dict(tickvals=[0, 1]),
                yaxis=dict(tickvals=[0, 1]),
                row=row_idx, col=col_idx
            )
            
        elif num_genes_to_plot == 3:
            # For 3 genes with fitness as color
            all_states = list(itertools.product([0, 1], repeat=3))
            x = [state[0] for state in all_states]
            y = [state[1] for state in all_states]
            z = [state[2] for state in all_states]
            
            # Calculate fitness for each state
            fitnesses = []
            for g1, g2, g3 in all_states:
                # For each state, generate and average multiple random configurations of other genes
                avg_fitness = 0
                num_samples = 50 if N > 3 else 1
                for _ in range(num_samples):
                    # Generate random values for other N-3 genes
                    if N > 3:
                        other_genes = np.random.RandomState(seed).randint(0, 2, N - 3)
                        genome = np.array([g1, g2, g3, *other_genes])
                    else:
                        genome = np.array([g1, g2, g3])
                    avg_fitness += model.calculate_fitness(genome)
                fitnesses.append(avg_fitness / num_samples)
            
            # Create scatter3d trace
            scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=10,
                    color=fitnesses,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Fitness")
                ),
                text=[f'Genome: [{g1}, {g2}, {g3}]<br>Fitness: {fitness:.4f}' 
                      for (g1, g2, g3), fitness in zip(all_states, fitnesses)],
                hoverinfo='text',
                name=f'K={K}'
            )
            
            # Add edges between points that differ by one bit
            edges_x = []
            edges_y = []
            edges_z = []
            
            for i, state1 in enumerate(all_states):
                for j, state2 in enumerate(all_states):
                    if sum(abs(np.array(state1) - np.array(state2))) == 1:
                        edges_x.extend([state1[0], state2[0], None])
                        edges_y.extend([state1[1], state2[1], None])
                        edges_z.extend([state1[2], state2[2], None])
            
            edges = go.Scatter3d(
                x=edges_x,
                y=edges_y,
                z=edges_z,
                mode='lines',
                line=dict(color='gray', width=1),
                hoverinfo='none',
                showlegend=False
            )
            
            # Calculate row and column for this subplot
            row_idx = plot_idx // num_cols + 1  # 1-indexed
            col_idx = plot_idx % num_cols + 1   # 1-indexed
            
            # Add traces to the figure
            fig.add_trace(scatter, row=row_idx, col=col_idx)
            fig.add_trace(edges, row=row_idx, col=col_idx)
            
            # Update axis labels
            fig.update_scenes(
                xaxis_title="Gene 1",
                yaxis_title="Gene 2",
                zaxis_title="Gene 3",
                xaxis=dict(tickvals=[0, 1]),
                yaxis=dict(tickvals=[0, 1]),
                zaxis=dict(tickvals=[0, 1]),
                row=row_idx, col=col_idx
            )
    
    # Update layout
    fig.update_layout(
        title_text=f"NK Model Fitness Landscape (N={N})",
        height=600 * num_rows,  # Adjust height based on number of rows
        width=1200 if num_plots > 1 else 800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig

def create_3d_landscape_with_adaptive_walk(N, K_values, seed=42):
    """
    Create interactive 3D visualizations of fitness landscapes with an adaptive walk path
    
    Args:
        N (int): Total number of genes
        K_values (list): List of K values to visualize
        seed (int): Random seed for reproducibility
        
    Returns:
        plotly.graph_objects.Figure: Interactive 3D plot figure
    """
    # For adaptive walk, we'll use a smaller N to make it more visible
    small_N = min(N, 3)
    
    # Filter K values to ensure they are less than small_N
    valid_K_values = [k for k in K_values if k < small_N]
    if not valid_K_values:
        valid_K_values = [0, small_N-1]  # Default to 0 and N-1 if no valid K values
    
    num_plots = len(valid_K_values)
    
    # Dynamically adjust the number of rows and columns based on the number of plots
    num_cols = min(3, num_plots)  # Maximum 3 columns per row
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate required rows
    print(f"Creating {num_rows} rows and {num_cols} columns for {num_plots} plots.")
    
    # Create specs for each subplot
    specs = [[{'type': 'scene'} for _ in range(num_cols)] for _ in range(num_rows)]
    
    # Create subplot titles
    subplot_titles = [f'K={K} with Adaptive Walk' for K in valid_K_values]
    
    # Create subplot figure
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        specs=specs,
        subplot_titles=subplot_titles
    )
    
    for plot_idx, K in enumerate(valid_K_values):
        # Create model with specified K value
        model = NKModel(N=small_N, K=K, seed=seed)
        
        # Generate all possible genomes
        all_genomes = list(itertools.product([0, 1], repeat=small_N))
        
        # Calculate fitness for each genome
        fitnesses = [model.calculate_fitness(genome) for genome in all_genomes]
        
        # Create 3D coordinates for genomes
        if small_N == 3:
            x = [genome[0] for genome in all_genomes]
            y = [genome[1] for genome in all_genomes]
            z = [genome[2] for genome in all_genomes]
            
            # Create scatter3d for genomes
            scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=[10 + 20*f for f in fitnesses],
                    color=fitnesses,
                    colorscale='Viridis',
                    opacity=0.7,
                    colorbar=dict(title="Fitness")
                ),
                text=[f'Genome: {genome}<br>Fitness: {fitness:.4f}' 
                      for genome, fitness in zip(all_genomes, fitnesses)],
                hoverinfo='text',
                name=f'Genomes K={K}'
            )
            
            # Add edges between points that differ by one bit
            edges_x = []
            edges_y = []
            edges_z = []
            
            for i, g1 in enumerate(all_genomes):
                for j, g2 in enumerate(all_genomes):
                    if sum(abs(np.array(g1) - np.array(g2))) == 1:
                        edges_x.extend([g1[0], g2[0], None])
                        edges_y.extend([g1[1], g2[1], None])
                        edges_z.extend([g1[2], g2[2], None])
            
            edges = go.Scatter3d(
                x=edges_x,
                y=edges_y,
                z=edges_z,
                mode='lines',
                line=dict(color='gray', width=1),
                hoverinfo='none',
                showlegend=False
            )
            
            # Calculate row and column for this subplot
            row_idx = plot_idx // num_cols + 1  # 1-indexed
            col_idx = plot_idx % num_cols + 1   # 1-indexed
            
            # Add traces to the figure
            fig.add_trace(scatter, row=row_idx, col=col_idx)
            fig.add_trace(edges, row=row_idx, col=col_idx)
            
        elif small_N == 2:
            x = [genome[0] for genome in all_genomes]
            y = [genome[1] for genome in all_genomes]
            z = fitnesses
            
            # Create scatter3d for genomes
            scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=12,
                    color=z,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Fitness")
                ),
                text=[f'Genome: {genome}<br>Fitness: {fitness:.4f}' 
                      for genome, fitness in zip(all_genomes, fitnesses)],
                hoverinfo='text',
                name=f'Genomes K={K}'
            )
            
            # Create surface connecting the points
            x_grid, y_grid = np.meshgrid([0, 1], [0, 1])
            z_grid = np.array(fitnesses).reshape(2, 2)
            
            surface = go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale='Viridis',
                opacity=0.5,
                showscale=False,
            )
            
            # Calculate row and column for this subplot
            row_idx = plot_idx // num_cols + 1  # 1-indexed
            col_idx = plot_idx % num_cols + 1   # 1-indexed
            
            # Add traces to the figure
            fig.add_trace(scatter, row=row_idx, col=col_idx)
            fig.add_trace(surface, row=row_idx, col=col_idx)
        
        # Simulate an adaptive walk from a random starting point
        start_idx = 0  # Start from genome [0, 0, 0] or [0, 0]
        current_genome = all_genomes[start_idx]
        walk_path = [current_genome]
        
        # Greedy hill climbing until reaching a local optimum
        while True:
            # Get all neighbors (1-bit mutations)
            neighbors = []
            for i in range(small_N):
                neighbor = list(current_genome)
                neighbor[i] = 1 - neighbor[i]  # Flip bit
                neighbors.append(tuple(neighbor))
            
            # Find the fittest neighbor
            neighbor_fitnesses = [model.calculate_fitness(n) for n in neighbors]
            best_neighbor_idx = np.argmax(neighbor_fitnesses)
            best_neighbor = neighbors[best_neighbor_idx]
            best_fitness = neighbor_fitnesses[best_neighbor_idx]
            
            # Stop if no improvement
            if best_fitness <= model.calculate_fitness(current_genome):
                break
                
            # Move to best neighbor
            current_genome = best_neighbor
            walk_path.append(current_genome)
        
        # Plot the adaptive walk path
        walk_x = [g[0] for g in walk_path]
        walk_y = [g[1] for g in walk_path]
        
        if small_N == 3:
            walk_z = [g[2] for g in walk_path]
            
            # Create line trace for the walk path
            walk_trace = go.Scatter3d(
                x=walk_x,
                y=walk_y,
                z=walk_z,
                mode='lines+markers',
                line=dict(color='red', width=6),
                marker=dict(
                    size=8,
                    color='red',
                ),
                name='Adaptive Walk'
            )
            
            fig.add_trace(walk_trace, row=row_idx, col=col_idx)
            
            # Update axis labels
            fig.update_scenes(
                xaxis_title="Gene 1",
                yaxis_title="Gene 2",
                zaxis_title="Gene 3",
                xaxis=dict(tickvals=[0, 1]),
                yaxis=dict(tickvals=[0, 1]),
                zaxis=dict(tickvals=[0, 1]),
                row=row_idx, col=col_idx
            )
            
        elif small_N == 2:
            walk_z = [model.calculate_fitness(g) for g in walk_path]
            
            # Create line trace for the walk path
            walk_trace = go.Scatter3d(
                x=walk_x,
                y=walk_y,
                z=walk_z,
                mode='lines+markers',
                line=dict(color='red', width=6),
                marker=dict(
                    size=8,
                    color='red',
                ),
                name='Adaptive Walk'
            )
            
            fig.add_trace(walk_trace, row=row_idx, col=col_idx)
            
            # Update axis labels
            fig.update_scenes(
                xaxis_title="Gene 1",
                yaxis_title="Gene 2",
                zaxis_title="Fitness",
                xaxis=dict(tickvals=[0, 1]),
                yaxis=dict(tickvals=[0, 1]),
                row=row_idx, col=col_idx
            )
            
    # Update layout
    fig.update_layout(
        title_text=f"NK Model Fitness Landscape with Adaptive Walk (N={small_N})",
        height=600 * num_rows,  # Adjust height based on number of rows
        width=1200 if num_plots > 1 else 800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig


# Main execution
if __name__ == "__main__":
    # Parameters
    N = 11  # Number of genes
    K_values = [0, 1, 2, 3, 4, 5 , 6, 7, 8, 9]  # Different K values to analyze
    
    # Analyze ruggedness for different K values
    print(f"Analyzing NK model with N={N} and K values {K_values}")
    results = analyze_NK_model_ruggedness(N, K_values)
    
    # Plot results
    fig1 = plot_ruggedness_analysis(results)
    plt.show()

    # For visualization, use a smaller N
    vis_N = 10
    vis_K_values = [0, 2, vis_N-1]
    
    print(f"Visualizing landscapes with N={vis_N} and K values {vis_K_values}")
    fig2 = visualize_multiple_landscapes(vis_N, vis_K_values)
    plt.show()
    
    
    # Explanation of epistasis
    print("\nEpistasis Explanation:")
    print("----------------------")
    print("Epistasis refers to the phenomenon where the effect of one gene mutation")
    print("is influenced by the presence or absence of mutations in other genes.")
    print("In the NK model, K controls the degree of epistatic interactions:")
    print("- K=0: No epistasis - each gene contributes independently to fitness")
    print("- K>0: Epistatic interactions - a gene's contribution depends on K other genes")
    print("- K=N-1: Maximum epistasis - each gene interacts with all other genes")
    print()
    print("As K increases, the fitness landscape becomes more rugged with many local")
    print("optima, making it harder for evolutionary processes to find the global")
    print("optimum. This occurs because changing one gene affects the fitness")
    print("contributions of K other genes, creating complex interdependencies that")
    print("result in a more complex, multi-peaked landscape.")

    # fig1 = create_3d_fitness_landscape(N, K_values, num_genes_to_plot=2)
    # fig1.show()

    # fig2 = create_3d_fitness_landscape(N, K_values, num_genes_to_plot=3)
    # fig2.show()

    # fig3 = create_3d_landscape_with_adaptive_walk(3, K_values)
    # fig3.show()