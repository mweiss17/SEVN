import gym
from SEVN_gym.envs.SEVN_base import SEVNBase
import numpy as np
from SEVN_gym.envs.utils import denormalize_image, unconvert_house_numbers, unconvert_street_name
from SEVN_gym.envs.wrappers import unwrap_obs
import cv2


def show_img(frame):
    img = np.swapaxes(frame[:3, :, :], 0, 2)
    img = cv2.resize(denormalize_image(img)[:, :, ::-1], (500, 500))
    cv2.imshow('SEVN viewer', img)
    cv2.waitKey(1)


def debug_output(obs):
    show_img(obs)
    data = unwrap_obs(obs, True, True, None, True, env.unwrapped.num_streets)
    print("goal hn", unconvert_house_numbers(data["goal_house_numbers"]))
    for i in range(3):
        print(f"visible {i}",
              unconvert_house_numbers(data["visible_house_numbers"][i]))

    print(
        "goal street",
        unconvert_street_name(data["goal_street_names"],
                              env.unwrapped.all_street_names))
    for i in range(2):
        print(
            f"visible street {i}",
            unconvert_street_name(data["visible_street_names"][i],
                                  env.unwrapped.all_street_names))


env = gym.make("SEVN-Train-AllObs-Shaped-v1")
print(env.unwrapped.all_street_names)

while True:
    print ("= = = RESETTING = = =")
    obs, done = env.reset(), False
    debug_output(obs)

    while not done:
        action = None
        while action not in ["a", "q", "w", "e", "d"]:
            action = input("your action (out of [aqwed]):")

        if action == "a":
            action = SEVNBase.Actions.LEFT_BIG
        elif action == "q":
            action = SEVNBase.Actions.LEFT_SMALL
        elif action == "w":
            action = SEVNBase.Actions.FORWARD
        elif action == "e":
            action = SEVNBase.Actions.RIGHT_SMALL
        elif action == "d":
            action = SEVNBase.Actions.RIGHT_BIG

        obs, rew, done, misc = env.step(action)
        debug_output(obs)
        print(f"shortest path length: {env.prev_spl}")
        print(f"Rew {rew}, done {done}, misc {misc}")
        print("=========")
