# Meshnet

Meshnet is a Graph Neural Network (GNN) model for the prediction of mesh parameters given a CAD model. The framework is based on the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) library. The network is based on the [MeshGraphNets](https://arxiv.org/abs/2010.03409) architecture.

## Setup

### Conda environment
```bash
mamba env create -f utils/envs/meshnet_no_builds.yml
conda activate meshnet
```

### Download data
The dataset is available. In a different folder than the one containing this repository, run:
```bash
git clone https://github.com/UgoPelissier/dataset.git
```
And follow the instructions in the README.md file. This will create `geo` and `geo_unrolled` folders inside `stokes2`, `stokes3` and `stokes3adapt` folders containing the CAD models.

Last step is to move the `stokes2`, `stokes3` and `stokes3adapt` folders inside the `data` folder of this meshnet repository. The final structure should look like this:

```
├── callbacks
├── configs
├── data
└── data
    └── stokes2
    ├── stokes3
    ├── stokes3adapt
        └── geo
            ├── cad_000.geo
            :
            └── cad_500.geo
        └── raw
            ├── cad_000.geo_unrolled
            :
            └── cad_500.geo_unrolled
    ├── datamodule.py
    └── dataset.py
├── model
├── utils
├── __init__.py
├── .gitignore
├── main.py
└── README.md
```

### Train the model
Parameters setting is done in the `configs/config.yaml` file. Check, and change if needed, the parameters marked with `# TODO` comments.

Train the model by running:
```bash
python main.py fit -c configs/config.yaml
```

You can get help on the command line arguments by running:
```bash
python main.py fit --help
```

It will create a new folder in the `logs/` folder containing the checkpoints of the model and a configuration file containing the parameters used for the training, that you can use later if you want.
### Evaluate the model
To evaluate the model training, run:
```bash
tensorboard --logdir=logs/
```

### Test the model
To test the model, run:
```bash
python main.py test -c configs/config.yaml --ckpt_path $ckpt_path
```
where `$ckpt_path` is the path to the checkpoint file located in the `logs/version_$version/checkpoints/` folder.

It will create a new folder in the `logs/` folder containing the meshes resulting from the predictions of the model (`vtk` files).

## Contact

\<[ugo.pelissier@etu.minesparis.psl.eu](mailto:ugo.pelissier@etu.minesparis.psl.eu)\>