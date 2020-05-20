# Accelerating data preprocessing for catalysis research

## Brief overview

We are focused on generating datasets and machine learning models for chemistry applications. Specifically, we are interested in catalysis.

## The problem

Currently, we are using models such as crystal graph convolutional neural network (CGCNN) that require converting our molecular structures to graph representations for integration with torch geometric and Pytorch. One of the slow operations involved in the conversion is a nearest neighbor search between atoms. The current implementation is fairly slow (50,000 examples ~ 20 mins), but this is manageable because the total amount of data we have is small; However, we are working on a dataset of nearly 1M examples where accelerating the preprocessing will be necessary. Theoretically, you only need to preprocess the data once, but in practice we do this operation a lot because we are constantly trying different features/representations.

Outside of the training data, we also want to make predictions on a massive number of configurations (~20M), which all require converting a structure to a graph. With these numbers in mind, I think a reasonable goal would be to speed up the algorithm by ~ 10x and additionally explore parallelization/chunking schemes.

## Accelerating nearest neighbor search

### Current implementation flow

ASE atoms object -> Pymatgen structure object -> structure.get_all_neighbors() -> returns distances and indexes of atoms within a given radius

### Ideas

1. Use a different Pymatgen method structure.get_neighbor_list() that utilizes cython. I am not sure entirely what algorithm the new method uses but there are a number recent pull requests that likely explain it. Preliminary timing results for the new Pymatgen method show ~3x speed up and are in the compare_pymatgen_methods notebook.

2. Use a GPU based nearest neighbor search (e.g. KD-tree algorithm), if we choose this approach, we can probably circumvent using Pymatgen altogether and just work off ASE atoms objects.

## Code

There is an example notebook called test_convert and a toy dataset in the data folder to experiment with. Please feel free to add notebooks or scripts to this folder and we will determine how to merge with the master branch later.

## Dependencies for preprocessing

- Pytorch
- Torch Geometric
- Pymatgen
- ASE

***version info can be found in env.cpu.yml
