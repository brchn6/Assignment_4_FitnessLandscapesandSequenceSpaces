# NK Model Fitness Landscape Visualization

This project implements the NK model to explore how epistatic interactions shape fitness landscapes in binary genomes. By adjusting the K parameter, we can tune landscape ruggedness and analyze evolutionary dynamics.

## The NK Model Implementation

### Binary Genome Representation
- Each genome is represented as a binary string of length N (where each gene can be either 0 or 1)
- The model has two key parameters:
  - **N**: Number of genes in the genome
  - **K**: Number of other genes each gene interacts with (0 ≤ K < N)

### Fitness Calculation
The implementation computes fitness as follows:
1. For each gene position i:
   - A random subset of K other genes is selected to interact with gene i
   - A contribution table is generated, mapping all possible states of gene i and its K interacting genes to fitness values
2. For a given genome:
   - Each gene's fitness contribution is determined by its state and the states of its K interacting genes
   - Overall fitness is calculated as the average of all individual gene contributions

### Core Components
- `generate_random_interaction_matrix()`: Creates matrices defining which genes interact with each other
- `generate_contribution_tables()`: Creates random fitness contribution tables for each gene
- `calculate_fitness()`: Computes the fitness of a specific genome
- `generate_fitness_landscape()`: Explores all possible genomes to map the complete landscape
- `find_local_optima()`: Identifies all fitness peaks (genomes with higher fitness than all neighbors)

## Tunable Landscape Ruggedness

### Visualization of K's Effect on Landscapes
The project includes multiple visualization tools:
- **2D Heatmaps**: Shows fitness projections for selected gene pairs across different K values
- **3D Surface Plots**: Visualizes fitness as a function of two genes
- **3D Interactive Plots**: Enables exploration of the fitness landscape with interactive controls
- **Adaptive Walk Visualization**: Shows evolutionary trajectories through the landscape

### Key Findings

Based on the visualizations shown in the images:

1. **At K=0 (No Epistasis)**:
   - The landscape is smooth with a single optimum
   - Gene contributions are independent
   - Fitness changes predictably when genes are mutated
   - Heatmap shows a simple gradient pattern

2. **At K=2 (Moderate Epistasis)**:
   - The landscape begins to show complexity
   - Multiple local optima emerge
   - Some gene combinations produce unexpected fitness values
   - Heatmap shows non-uniform fitness distribution

3. **At K=9 (High Epistasis)**:
   - The landscape becomes highly rugged
   - Many local optima appear (approximately 140 for N=10)
   - Fitness becomes difficult to predict from individual genes
   - Heatmap shows complex patterns with multiple peaks and valleys

## Calculations and Analysis

### Quantification of Ruggedness
The left graph in Image 1 quantifies how landscape ruggedness increases with K:
- **K=0**: ~1 local optimum (smooth landscape)
- **K=2**: ~8 local optima
- **K=4**: ~30 local optima
- **K=6**: ~65 local optima
- **K=8**: ~115 local optima
- **K=9**: ~140 local optima

This exponential relationship demonstrates how higher K values create increasingly complex fitness landscapes.

### Average Fitness Analysis
The right graph in Image 1 shows average fitness across different K values:
- Average fitness remains relatively stable around 0.5 regardless of K
- There is a slight dip around K=2 (to approximately 0.47)
- Error bars indicate variability across multiple simulation runs
- As K approaches N-1, average fitness stabilizes

This suggests that while landscape topology changes dramatically with K, the average fitness of random genomes remains consistent.

## Explanation of Epistasis

Epistasis occurs when the effect of one gene mutation depends on the genetic background (the states of other genes). In the NK model:

- **Without epistasis (K=0)**: Each gene contributes independently to fitness. Changing one gene always has the same effect regardless of other genes' states. This creates a smooth, single-peaked landscape that is easy for evolution to navigate.

- **With epistasis (K>0)**: A gene's contribution depends on K other genes. The same mutation can be beneficial, neutral, or detrimental depending on the states of interacting genes. This creates a rugged landscape with multiple peaks and valleys.

- **At maximum epistasis (K=N-1)**: Each gene interacts with all other genes. The fitness landscape becomes extremely rugged with many local optima, making it difficult for evolutionary processes to find the global optimum through gradual mutations.

Higher K leads to more rugged landscapes because:
1. Each gene's fitness contribution depends on more genes
2. Changing one gene affects the fitness contributions of K other genes
3. This creates complex interdependencies that result in multiple fitness peaks separated by valleys

The fitness projections in Image 2 clearly demonstrate this increasing complexity. At K=0, the landscape forms a simple gradient, while at K=9, we see a complex pattern of high and low fitness regions with no clear path between them.

## Usage Example

```python
# Basic example
import numpy as np
from nk_model import (
    NKModel, 
    create_3d_fitness_landscape_interactive,
    create_3d_landscape_with_adaptive_walk_interactive,
    analyze_NK_model_ruggedness,
    plot_ruggedness_analysis
)

# Set parameters
N = 10  # Number of genes
K_values = [0, 2, 4, 6, 8, 9]  # Different K values to analyze

# Analyze ruggedness
results = analyze_NK_model_ruggedness(N, K_values, num_runs=5)
fig1 = plot_ruggedness_analysis(results)
fig1.show()

# Create 2D fitness landscape visualization
fig2 = create_3d_fitness_landscape_interactive(N, K_values, num_genes_to_plot=2)
fig2.show()

# Create 3D fitness landscape visualization
fig3 = create_3d_fitness_landscape_interactive(N, K_values, num_genes_to_plot=3)  
fig3.show()

# Create adaptive walk visualization
small_N = 3  # Smaller N for clearer visualization
valid_K_values = [0, 1, 2]  # Must be < small_N
fig4 = create_3d_landscape_with_adaptive_walk_interactive(small_N, valid_K_values)
fig4.show()
```

## Interactive Visualization Features

The toolkit provides several interactive visualization tools:

1. **2D Fitness Projection**: Visualizes how the average fitness changes across different combinations of two genes, with the states of other genes averaged
   
2. **3D Fitness Landscape**: Shows fitness as a function of two genes, with the z-axis representing fitness

3. **3D Genotype Space**: Represents genomes as points in 3D space, with color indicating fitness values

4. **Adaptive Walk Visualization**: Simulates evolutionary paths starting from a specific genome and using a greedy hill-climbing algorithm to reach a local optimum

## Interaction Tips

1. **Rotation**: Click and drag to rotate the 3D visualizations
2. **Zoom**: Use the scroll wheel to zoom in and out
3. **Pan**: Right-click and drag to pan
4. **Reset View**: Double-click to reset the view
5. **Information**: Hover over data points to see detailed information
6. **Isolation**: Click on legend items to isolate specific traces

## Requirements

- Python 3.7+
- NumPy
- Plotly
- Seaborn (for static visualizations)
- Matplotlib (for static visualizations)
- tqdm (for progress bars)

## References

- Kauffman, S. A. (1993). The Origins of Order: Self-Organization and Selection in Evolution. Oxford University Press.
- Kauffman, S., & Levin, S. (1987). Towards a general theory of adaptive walks on rugged landscapes. Journal of Theoretical Biology, 128(1), 11-45.
- Wright, S. (1932). The roles of mutation, inbreeding, crossbreeding, and selection in evolution. Proceedings of the Sixth International Congress of Genetics, 1, 356-366.