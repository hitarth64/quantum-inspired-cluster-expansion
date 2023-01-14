# Quantum-inspired Cluster Expansion

Quantum-inspired Cluster Expansion (or QCE in short) is a mapping of classical cluster expansion to Quadratic Unconstrained Binary Optimization (QUBO) expression or in general, an Ising model. This provides a very efficient way to search the large combinatorial chemical space. QCE can be treated both as a standalone ```surrogate+search``` method or as a standalone ```search``` method. 

- For its use as ```surrogate+search``` method, one needs to use DFT data to train cluster expansions and then transfer it to QUBO format using QCE

- For its use as a ```search``` method, one can feed use data generated using ML-potential such as the one based on [OCP](https://opencatalystproject.org/) and then perform QCE using the generated data. Data generation and relaxtion using such an ML potential for large number of structures can be computationally expensive. Thus, when combined with QCE, can enable exhaustive search.

Following paper describes the details of QCE framework: [Accelerated Chemical Space Search using Quantum-inspired Cluster Expansion](https://arxiv.org/abs/2205.09007)

## Table of contents:

- [Install](#how-to-install)
- [Requirements](#requirements)
- [Usage](#usage)
- [Data](#data)
- [How to cite](#how-to-cite)
- [Authors](#authors)
- [License](#license)

## How to install
You can install the latest version of QCE from github as: 

```pip install git+https://github.com/hitarth64/quantum-inspired-cluster-expansion```

##  Requirements
This package requires:
- [ICET](https://icet.materialsmodeling.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [ase](https://wiki.fysik.dtu.dk/ase/index.html)

Depending on the engines of interest, you may also want to install:
- [DEAP](https://deap.readthedocs.io/en/master/)
- [scikit-optimize](https://scikit-optimize.github.io/stable/)

Note: Both the required as well as optional dependencies are installed if you follow the instructions above for [installation](#how-to-install)

## Data
All the datasets generated and used in the paper and/or examples here are present in the current repository. 
You can access the data through the ```data``` module of ```qce``` as ```from QuantumInspiredClusterExpansion import data```
Please refer to the [dataset](QuantumInspiredClusterExpansion/dataset) page for more information.

If you are looking for the data on slabs generated, please refer to them at [MixedMetalOxides](https://github.com/hitarth64/MixedMetalOxides)

## Usage
We provide several examples in the ```examples``` directory. Directory has both notebooks as well as python files. 

For our study, we used Digital Annealer which is a commercial system. However, we provide complete functionality through QCE to extract the QUBO formulations that can be tested on various platforms including quantum annealers, quantum-inspired platforms and heursitic solvers. We also provide options to interface the search problem with various Genetic algorithms (GA) through DEAP as well as Bayesian Optimization (BO) strategies through scikit-learn. 

## How to cite
Please cite the following work if you use QCE.
```
@article{Choubisa2022,
   author = {Hitarth Choubisa and Jehad Abed and Douglas Mendoza and Hidetoshi Matsumura and Masahiko Sugimura and Zhenpeng Yao and Ziyun Wang and Brandon R. Sutherland and Alán Aspuru-Guzik and Edward H. Sargent},
   doi = {10.1016/J.MATT.2022.11.031},
   issn = {2590-2385},
   journal = {Matter},
   month = {12},
   publisher = {Cell Press},
   title = {Accelerated chemical space search using a quantum-inspired cluster expansion approach},
   url = {https://linkinghub.elsevier.com/retrieve/pii/S2590238522006622},
   year = {2022},
}

```

## License

This package has been released under BSD 3-Clause Clear License. Refer to [license](LICENSE) for details.

## Authors

The package has been co-written by Hitarth Choubisa from University of Toronto and Hidetoshi Matsumura from Fujitsu Consulting(Canada) Inc.
