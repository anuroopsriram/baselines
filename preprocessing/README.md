# Accelerating data preprocessing for catalysis research

## Brief overview

We are focused on generating datasets and machine learning models for chemistry applications. Specifically, we are interested in catalysis.

## The problem

Currently, we are using models such as crystal graph convolutional neural network (CGCNN) that require converting our molecular structures to graph representations for integration with torch geometric and Pytorch. One of the slow operations involved in the conversion is a nearest neighbor search between atoms. The current implementation is fairly slow (50,000 examples ~ 20 mins), but this is manageable because the total amount of data we have is small; However, we are working on a dataset of nearly 1M examples where accelerating the preprocessing will be necessary. Theoretically, you only need to preprocess the data once, but in practice we do this operation a lot because we are constantly trying different features/representations.

## Accelerating nearest neighbor search

### Current implementation flow

ASE atoms object -> Pymatgen structure object -> structure.get_all_neighbors() -> returns distances and indexes of atoms within a given radius

### Ideas

1. Use a different Pymatgen method structure.get_neighbors_list() that utilizes cython. I am not entirely what algorithm the new method uses but there are a number recent pull requests that likely explain it. This is super easy and I will make a comparison soon.

2. Use a GPU based KD-tree algorithm or similar, if we choose this approach, we could probably circumvent using Pymatgen altogether and just work off ASE atoms objects.

## Code

There is an example notebook called test_convert and a toy dataset in the data folder to experiment with. Please feel free to add testing notebooks or scripts to this folder (preprocessing) and we will determine how to merge with the master branch later.

## Dependencies for preprocessing:

- Pytorch
- Torch Geometric
- Pymatgen
- ASE

*** versions can be found in evn.cpu.yml
