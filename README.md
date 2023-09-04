# Meshnet

## Description

Meshnet is a Graph Neural Network (GNN) model for the prediction of mesh parameters given a CAD model. The framework is based on the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) library. The network is based on the [MeshGraphNets](https://arxiv.org/abs/2010.03409) architecture.

## Setup @Safran™

### Conda environment
```bash
module load conda
module load artifactory
mamba env create -f utils/envs/meshnet.yml
conda activate meshnet
```

### Download data
The data is available on the Safran GitLab. In a different folder than the one containing this repository, run:
```bash
git clone https://github.com/UgoPelissier/dataset
```
And follow the instructions in the README.md file. This will create `geo` folders inside `stokes2` and `stokes3` folders containing the CAD models.

Last step is to move the `stokes2` and `stokes3` folders inside the `data` folder of this meshnet repository. The final structure should look like this:

```
├── callbacks
├── configs
├── data
└── data
    └── stokes2
        └── raw
            └── geo
                ├── cad_000.geo
                :
                └── cad_500.geo
    ├── stokes3
        └── raw
            └── geo
                ├── cad_000.geo
                :
                └── cad_500.geo
    ├── datamodule.py
    └── dataset.py
├── model
├── utils
├── __init__.py
├── .gitignore
├── main.py
└── README.md
```

## Setup @Ext™

### Conda environment
```bash
mamba env create -f utils/envs/meshnet_no_builds.yml
conda activate meshnet
```

### Download data
Follow the same instructions as for the Safran setup except that the data is available on the GitHub repository:
```bash
git clone https://github.com/UgoPelissier/dataset.git
```

## Contact

Ugo Pelissier \
\<[ugo.pelissier.ext@safrangroup.com](mailto:ugo.pelissier.ext@safrangroup.com)\> \
\<[ugo.pelissier@etu.minesparis.psl.eu](mailto:ugo.pelissier@etu.minesparis.psl.eu)\>
