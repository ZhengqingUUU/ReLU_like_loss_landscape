## Loss Landscape of Shallow ReLU-like Neural Networks:  Stationary Points, Saddle Escaping, and Network Embedding



To see the results, you can run the following code:

(1) `training.ipynb`: It trains a network with very small initialization scale. This training process was the Figure 3 in the paper.

(2) `training_process_visualization.ipynb`: It produce Figure 3 in the paper.

(3) `perturbation_at_the_end.ipynb`: It perturbs the parameters obtained at the end of the training. It will show that the stationary point at the end is a local minimum, which corresponds to Figure 10 in the paper.

The code can be reused for other initialization scales.
