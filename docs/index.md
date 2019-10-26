---
layout: default
---

# Overview
In our endeavour to create a navigation assistant for the BVI, we found that existing RL environments were unsuitable for outdoor pedestrian navigation.
So we created SEVN, a sidewalk simulation environment and a neural network-based approach to creating a navigation agent. 
We hope that this dataset, simulator, and experimental results will provide a foundation for further research into the creation of agents that can assist members of the BVI community with outdoor navigation.

For a longer overview, please [read this post](/SEVN/01-article-env-introduction). For more examples, [see here](/SEVN/examples).

# SEVN Simulator [[code]](https://github.com/mweiss17/SEVN)
SEVN contains 4,988 full panoramic images and labels for house numbers, doors, and street name signs, which can be used for several different navigation tasks.
Agents trained with SEVN have access to variable-resolution images, visible text, and simulated GPS data to navigate the environment. 
The SEVN Simulator is OpenAI Gym-compatible to allow the use of state-of-the-art deep reinforcement learning algorithms.
An instance of the simulator using low-resolution imagery can be run at 400-800 frames per second on a machine with 2 CPU cores and 2 GB of RAM.

# SEVN Data Pipeline [[code]](https://github.com/mweiss17/SEVN-data)
Data pre-processing for SEVN (Sidewalk Simulation Environment for Visual Navigation). 
This takes raw 360° video as an input. The camera used was the Vuze+ 3D 360 VR Camera. 
The Vuze+ has four synchronized stereo cameras. 
Each stereo camera is composed of two image sensors with fisheye lenses that each capture full high definition video (1920x1080) at 30 Frames Per Second (FPS).

# The Model [[code]](https://github.com/mweiss17/SEVN-model)
In this repository you'll find the code used to train the multi-modal agents on SEVN. 
These agents can take in images, scene-text, and gps to navigate to goal addresses.
This repository was forked from [this PPO repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).

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





