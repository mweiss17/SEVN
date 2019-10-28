---
layout: default
---

# How to Create a Navigation Assistant for the Visually Impaired
## Step 1: Build a Sidewalk Simulation Environment


![](https://i.imgur.com/okfisip.jpg)

*As part of the NAVI project, we have released our dataset and environment called SEVN (Sidewalk Simulation Environment for the Visually Impaired). We hope that this work will enable future research in visual navigation and, in the long term, contribute to allowing the blind and visually impaired to gain autonomy when commuting.*


## NAVI: Navigational Assistant for the Visually Impaired

For a moment, imagine you will imminently become blind to a degenerative illness. You must find a [rehab center](https://www.cnib.ca/en/programs-and-services/rehab--services?region=qc) and get training to use a cane, learn to love Siri, and buy an Alexa. You're going to have to [learn how to be independent all over again](https://nfb.org/images/nfb/publications/bm/bm14/bm1404/bm140414.htm). Unfortunately, there are some compromises you'll have to make. Take this example.

You're at home on a Friday evening when you get a text inviting you to a party. But venturing to a new location is tough without the help of a sighted person. Sure, you can get out of the house, and Uber can take you most of the way, but then what? There are a lot of [tools](https://www.sciencedirect.com/science/article/pii/S1877705813016214) out there that can help, but most of them rely on GPS and can't actually get you all the way to the venue. The [perfect end-to-end solution](https://arxiv.org/pdf/1811.10120.pdf) does not yet exist.

That's why we're building a tool called NAVI, the Navigational Assistant for the Visually Impaired. NAVI uses GPS and vision -- even reading scene text like house numbers and street signs -- to dynamically navigate!

![](https://i.imgur.com/pn2EWu1.png)

There are three primary processes that make up the NAVI system:

 -  **Sensing:** Before even thinking of understanding our surroundings, we need our system to have access to raw information about its environment. Considering that we want our solution to be easily available, we assume the user only has access to the sensors that are included in a regular smartphone, such as a camera and a GPS. These signals need to be pre-processed to be used by the following  components.
- **Perception:** Once  we have access to all this information about our surroundings, we need to extract  relevant features from it. This includes any text that is visible in the scene, such as house numbers and street names, as well as any useful visual features.
- **Reasoning:** Finally, this is the most important part of the process. Given all relevant information about our surroundings and a precise goal destination, we need to make an informed decision about what actions to take in order to get there.

These are the processes that take place, usually subconsciously, for those that are not visually impaired – sensing, perception and reasoning. Our final goal is to recreate these processes in the NAVI technology. But before we can successfully develop this solution, there are three key components that have yet to be tackled.

1. Create a navigation environment with real-world data and simulates sensor signals
2. Train an RL agent to navigate that environment
3. Transfer that agent to an edge device for real-world use

We are committed to developing these systems as free, open-source software. Any future deployed NAVI system will _not_ upload your location to a server or track your behaviour _in any way_.

In our work [SEVN: A Sidewalk Simulation Environment for Visual Navigation](https://linktoourpaper), we present an RL environment to train agents in and set a baseline performance on the navigation task using the RL algorithm Proximal Policy Optimization.

## A Sidewalk Simulation Environment for Visual Navigation

###  Real World Data

Our dataset contains 5k+ panoramic images taken from Montreal's Little Italy. They cover 7 blocks, totaling 6.3km of sidewalk. We annotated doors, house numbers and street signs and provided ground truth text labels for each annotation.

![](https://i.imgur.com/JhUAn97.png)

<!--- We have also made available the pre-processed house number and street sign crops from our dataset, which could be used to test Optical Character Recognition recognition models :point_right: [link](https://linktotextcrops) --->


### An Open-AI gym environment

![](https://i.imgur.com/xRG7gDI.gif)

Our Sidewalk Simulation Environment for Visual Navigation (SEVN) is a realistic representation of what it is like to travel over the sidewalk in a typical urban neighborhood. The goal is to use this simulated environment to train Reinforcement Learning agents that will eventually be able to solve this task and transfer to the real world.

The simulator allows the agent to observe a part of a specific panorama based on its position and orientation, and navigate in the environment by rotating or transitioning to neighboring panoramas. Many tasks are available, they have different sensor modalities and present different challenges with varying difficulty.

We chose Montreal's Little Italy because it features a high variety of zones and encompasses interesting characteristics that make it a challenging setting for navigation. In a relatively small area, there are many different types of zones, such as residential commercial and industrial areas.


### How we built it

To build our environment, we collected raw video with a 360 degree [Vuze+](https://vuze.camera/camera/vuze-plus-camera/) mounted on a monopod and held sligthly above eye-height. The Vuze+ is equiped with four synchronized stereo cameras, each one composed of two images sensors with fisheye lenses. We captured 360 degree high definition video, from which we extracted 30 frames per second.

![](https://i.imgur.com/E5FyVFf.png)

To get an accurate localisation for each one of our frames, we used [ORBSLAM2](https://github.com/raulmur/ORB_SLAM2), a Simultaneous Localisation and Mapping pipeline. This allowed us to triangulate the position of the camera at every timestep relying only on visual features from our footage.

In parallel, we used the [VuzeVR Studio software](https://vuze.camera/vr-software/) to stitch together the raw footage from each camera and obtain a 360 stabilized video, from which we extracted 3840 x 2160 pixel equirectangular projections.

Finally, we used the coordinates from each image to geo-localise the 360 degree equirrectangular panoramas. We constructed a graph with these panoramas, where an agent can navigate by transferring between neighboring images. This results in a Google Street View--like environment. For a more detailed explanation of the process, take a look at our [paper](https://linktoourpaper) and our [code](https://github.com/simonchamorro/SEVN-data).

![](https://i.imgur.com/wdb0Jpc.jpg)


### How to use it

Our environment [SEVN](https://github.com/mweiss17/SEVN) is compatible with Open AI gym and quite easy to use. Here is an example:

```
import gym
import SEVN_gym

env = gym.make("SEVN-Mini-All-Shaped-v1")
obs = env.reset() # See obs 1
```

In *SENV-Mini-All-Shaped*, the agent has access to all different modalities. The observation contains:

- Image (84 x 84 pixels): The visible part of the current panorama, covering a horizontal FOV of 135 degrees.
- Mission: The goal adress is the combination of a street name and a house number. Street names are encoded as one-hot vectors of length 7, since there are 7 different streets in our dataset. House numbers are composed of 4 digits, and each one of those is encoded as a one-hot vector of length 10.
- Relative and Goal GPS: The relative GPS coordinates of the goal as well as its absolute coordinates.
- Visible Text: In this environment, the ground truth label from any text that is inside the agent's FOV is passed as part of the observation.

```
action = env.Actions['FORWARD']
obs, rew, done, info = env.step(action) # See obs 2
```

This environment features a shaped reward. The agent get a positive reward every time it gets closer to its goal, and a negative reward when it moves away from its goal.

```
action = env.Actions['BIG_RIGHT']
obs, rew, done, info = env.step(action) # See obs 2
```

![](https://i.imgur.com/8tUNV0L.png)

We hope that our environment will contribute to the progress of state-of-the-art Visual Navigation models, and will be a stepping stone for coming up with a system that will solve this problem for the blind and visually impaired.