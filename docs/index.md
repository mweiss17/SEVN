---
layout: default
---

## Overview
In our endeavour to create a navigation assistant for the BVI, we found that existing RL environments were unsuitable for outdoor pedestrian navigation.
So we created SEVN, a sidewalk simulation environment and a neural network-based approach to creating a navigation agent. 
We hope that this dataset, simulator, and experimental results will provide a foundation for further research into the creation of agents that can assist members of the BVI community with outdoor navigation.

## Dataset [[examples]](/SEVN/examples)
In this page you'll find few examples from the SEVN dataset and the simulator.
The dataset covers different types of regions from the city of Montreal, including residential, commercial and industrial.
The SEVN Simulator is available in low resolution and high resolution.

## Creating a Navigation Assistant for the Visually Impaired [[read more]](/SEVN/01-article-env-introduction)
### Step 1: Build a Sidewalk Simulation Environment

![](https://i.imgur.com/okfisip.jpg)

*As part of the NAVI project, we have released our dataset and environment called SEVN (Sidewalk Simulation Environment for the Visually Impaired). We hope that this work will enable future research in visual navigation and, in the long term, contribute to allowing the blind and visually impaired to gain autonomy when commuting.*

## SEVN Simulator [[code]](https://github.com/mweiss17/SEVN)
SEVN contains 4,988 full panoramic images and labels for house numbers, doors, and street name signs, which can be used for several different navigation tasks.
Agents trained with SEVN have access to variable-resolution images, visible text, and simulated GPS data to navigate the environment. 
The SEVN Simulator is OpenAI Gym-compatible to allow the use of state-of-the-art deep reinforcement learning algorithms.
An instance of the simulator using low-resolution imagery can be run at 400-800 frames per second on a machine with 2 CPU cores and 2 GB of RAM.

## SEVN Data Pipeline [[code]](https://github.com/mweiss17/SEVN-data)
Data pre-processing for SEVN (Sidewalk Simulation Environment for Visual Navigation). 
This takes raw 360° video as an input. The camera used was the Vuze+ 3D 360 VR Camera. 
The Vuze+ has four synchronized stereo cameras. 
Each stereo camera is composed of two image sensors with fisheye lenses that each capture full high definition video (1920x1080) at 30 Frames Per Second (FPS).

## SEVN Baseline Model [[code]](https://github.com/mweiss17/SEVN-model)
In this repository you'll find the code used to train the multi-modal agents on SEVN. 
These agents can take in images, scene-text, and gps to navigate to goal addresses.

## Paper [[Arxiv]](https://arxiv.org/abs/1910.13249)

If you use this work, please cite us. Here's the Bibtex for our paper.

```
@misc{weiss2019navigation,
    title={Navigation Agents for the Visually Impaired: A Sidewalk Simulator and Experiments},
    author={Martin Weiss and Simon Chamorro and Roger Girgis and Margaux Luck and
            Samira E. Kahou and Joseph P. Cohen and Derek Nowrouzezahrai and
            Doina Precup and Florian Golemo and Chris Pal},
    year={2019},
    eprint={1910.13249},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```