import gym
from SEVN_gym.envs.SEVN_base import SEVNBase
import numpy as np
from SEVN_gym.envs.utils import denormalize_image, unconvert_house_numbers, unconvert_street_name, Actions
from SEVN_gym.envs.wrappers import unwrap_obs
import cv2
import argparse


def show_img(env, frame):
    img = np.swapaxes(frame[:3, :, :], 0, 2)
    img = cv2.resize(denormalize_image(img)[:, :, ::-1], (1500, 1500), interpolation = cv2.INTER_NEAREST)
    cv2.imshow('SEVN viewer', img)
    return cv2.waitKey(-1)


def debug_output(env, obs):

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

    print(f"shortest path length: {env.unwrapped.prev_spl}")

def play(env, zoom=4):
    while True:
        print("= = = RESETTING = = =")
        obs, done = env.reset(), False
        print(
            "Highlight the Viewer window, press one of [aqwed]. Press 'x' to quit. Press 'r' to reset."
        )
        key = show_img(env, obs)

        while not done:
            import time
            time.sleep(.05)
            while key not in [ord(x) for x in ["a", "q", "w", "e", "d", "x", "r"]]:
                key = show_img(env, obs)

            reset = False
            if key == ord("a"):
                action = Actions.LEFT_BIG
            elif key == ord("q"):
                action = Actions.LEFT_SMALL
            elif key == ord("w"):
                action = Actions.FORWARD
            elif key == ord("e"):
                action = Actions.RIGHT_SMALL
            elif key == ord("d"):
                action = Actions.RIGHT_BIG
            elif key == ord("x"):
                print("quitting")
                quit()
            elif key == ord("r"):
                print("= requested resetting")
                reset = True
            key = None

            if not reset:
                obs, rew, _, info = env.step(action)
            else:
                obs = env.reset()

            debug_output(env, obs)

            if not reset:
                print(f"Rew {rew}, done {done}, info {info}")
            print("=========")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str, default='SEVN-Train-AllObs-Shaped-v1',
                        help='Define Environment')
    parser.add_argument('--high-res', action="store_true",
                        help='Use high-resolution images')
    args = parser.parse_args()
    if args.high_res:
        zoom = 0.5
    else:
        zoom = 4
    env = gym.make(args.env, high_res=args.high_res)
    play(env, zoom=zoom)

if __name__ == '__main__':
    main()