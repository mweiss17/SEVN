[![https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg](https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg)](https://singularity-hub.org/collections/3288)

# SEVN

SEVN: Sidewalk Simulator Environment for Visual Navigation. An outdoor environment simulator with real-world imagery for Deep Reinforcement Learning on navigation tasks.

![game.png](img/game.png)

## Requirements

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

pip install tensorflow

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..

# Install env
git clone https://github.com/mweiss17/SEVN.git
cd SEVN
pip install -e .
```

## To play
```
python scripts/01-play.py
```

## For more information
[Creating a Navigation Assistant for the Visually Impaired](https://github.com/mweiss17/SEVN/blob/master/docs/01-article-env-introduction.md)


## Things that are currently broken:

- scripts/01-play.py uses outdated `meta.hdf5`
- env.render("human") currently doesn't work, because the render function requires some "first_time" and "clear" parameters
