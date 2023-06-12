# Stage Ugo Pelissier

## Setup @Safran™

```bash
module load conda
module load artifactory
mamba env create -f utils/envs/meshnet.yml
conda activate meshnet
```

## Setup @Ext™

```bash
mamba env create -f utils/envs/meshnet_no_builds.yml
conda activate meshnet
```

## Contact

Ugo Pelissier \
\<[ugo.pelissier.ext@safrangroup.com](mailto:ugo.pelissier.ext@safrangroup.com)\> \
\<[ugo.pelissier@etu.minesparis.psl.eu](mailto:ugo.pelissier@etu.minesparis.psl.eu)\>
