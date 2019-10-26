---
layout: default
---

# Abstract
In our endeavour to create a navigation assistant for the BVI, we found that existing RL environments were unsuitable for outdoor pedestrian navigation.
This work introduces SEVN, a sidewalk simulation environment and a neural network-based approach to creating a navigation agent.
SEVN contains panoramic images with labels for house numbers, doors, and street name signs, and formulations for several navigation tasks.
We study the performance of an RL algorithm (PPO) in this setting. Our policy model fuses multi-modal observations in the form of variable resolution images, visible text, and simulated GPS data to navigate to a goal door. 
We hope that this dataset, simulator, and experimental results will provide a foundation for further research into the creation of agents that can assist members of the BVI community with outdoor navigation.

# SEVN Simulator 
SEVN contains 4,988 full panoramic images and labels for house numbers, doors, and street name signs, which can be used for several different navigation tasks. Agents trained with SEVN have access to variable-resolution images, visible text, and simulated GPS data to navigate the environment. The SEVN Simulator is OpenAI Gym-compatible to allow the use of state-of-the-art deep reinforcement learning algorithms. An instance of the simulator using low-resolution imagery can be run at 400-800 frames per second on a machine with 2 CPU cores and 2 GB of RAM.



# Paper 

If you use this work, please cite us:

```
Martin Weiss, Simon Chamorro, Roger Girgis, Margaux Luck, Samira Ebrahimi Kahou, 
Joseph Paul Cohen, Derek Nowrouzezahrai, Doina Precup, Florian Golemo, Chris Pal. 
"Navigation Agents for the Visually Impaired: A Sidewalk Simulator and Experiments" 
In Conference on Robot Learning. 2019.
```

Or via our bibtex:

(TODO insert Arxiv bibtex here)


## Code
[The Environment](https://github.com/mweiss17/SEVN)

[The Model](https://github.com/mweiss17/SEVN-model)

[The Data Pipeline](https://github.com/mweiss17/SEVN-data)


