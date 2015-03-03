### random_forest

This code is the implementation for a random forest in Python3.  It uses three 
forms of randomization, which is at the heart of the strength of random forests.
Randomization allows this algorithm to not overfit to the data as hundreds or
thousands of stumps or trees are added, and allows it to build a complex, 
nonlinear representation of the underlying true distribution of the class 
variables.  

#### Randomization

The first form of randomization used here is random subsampling of the data, or
what is known as bagging.  This is extremely useful when dealing with large 
datasets, and also helps the algorithm to learn.  

The second form of randomization is random feature selection.  That is, if the 
underlying data has hundreds or thousands of input features, at each level of
the stump or tree only a random subset of all the features will be considered 
when deciding on an optimal split point.

The final form of randomization comes from random selection of the split points
to use within a single feature.  That is, once the random features are selected,
a number of random split points (if using real valued data) can also be selected,
and some metric (such as entropy) evaluated at those split points to determine 
if this will be the optimal split decision to select at this level of the tree, 
given the sampled dataset and random features under consideration.

#### Parallelization

Random forests are inherently parallelizable.  They can be trained in parallel using
different bags of data, different random features, and different random split points.
This greatly reduces the amount of time required for training and testing the model.
Python's multiprocessing.pool is used in order to parallelize the creation of trees 
in the forest.
