# Periodic Level Set Function Example

This repository provides a minimum working example for the preprint / Scientific Reports submission "Level set methods for gradient-free optimization of metasurface arrays" [(Saad-Falcon et al.)](http://arxiv.org/abs/2307.07606).

This example assumes the use of the conda package manager on a `linux-64` or `osx-64` platform. To recreate the environment, `cd` into this directory and run the command

```
conda env create -f environment.yml
conda activate lsf-mwe
```

The example can then be run with

```
python model.py
```

which should produce a figure containing transmission and reflection magnitudes for a level set function surface.
