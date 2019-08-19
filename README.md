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
