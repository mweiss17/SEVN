import time

import gym
from tqdm import trange

import SEVN_gym
import matplotlib.pyplot as plt
import numpy as np
from SEVN_gym.envs.utils import denormalize_image, unconvert_house_numbers, unconvert_street_name
from SEVN_gym.envs.wrappers import unwrap_obs


env = gym.make("SEVN-Train-AllObs-Shaped-v1")

timings = []


##### IN ORDER FOR THIS TO WORK, YOU NEED TO ADD THE TIMING CODE BACK INTO THE BASE CLASS
# start = time.time()
# ## block of code
# self.diff_a.append(time.time() - start)


for i in range(10):
    start = time.time()

    obs, done = env.reset(), False
    # debug_output(obs)

    for j in range(1000):
        # while not done:
        #     in_ = input("your action (out of [aqwed]):")
        env.step(env.action_space.sample())

        if (j+1) % 10 == 0:
            print ("===")
            for diff in ["a","b","c","d","e","f"]:
            # for diff in ["a","b","c","d"]:
                timings = getattr(env.unwrapped, f"diff_{diff}")
                print(
                    f"avg time for {diff}: {np.mean(timings)}s, i.e. {1 / np.mean(timings)}Hz"
                )

    diff = time.time() - start
    timings.append(diff)


