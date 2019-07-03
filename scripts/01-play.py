from collections import deque
import argparse
import time
import matplotlib
import gym
import hyrule_gym
from gym import logger
import pygame
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
    plt = None
from pygame.locals import VIDEORESIZE

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

def display_text(screen, text, video_size):
    textSurface = pygame.font.Font('freesansbold.ttf',30).render(text, True, (0,0,0))
    textRect = textSurface.get_rect()
    textRect.center = (video_size[0]/10, video_size[1]/10)
    screen.blit(textSurface, textRect)
    pygame.display.update()

def display_bb(screen, rel_coords, text, video_size):
    textSurface = pygame.font.Font('freesansbold.ttf',30).render(text, True, (0,0,0))
    textRect = textSurface.get_rect()
    BLUE=(0,0,255)
    pygame.draw.rect(screen, BLUE, rel_coords)
    textRect.center = (video_size[0]/10, video_size[1]/10)
    screen.blit(textSurface, textRect)
    pygame.display.update()

def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    """Allows one to play the game using keyboard.

    To simply play the game use:

        play(gym.make("Hyrule-v0"))

    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.

    If you wish to plot real time statistics as you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.

        def callback(obs_t, obs_tp1, action, rew, done, info):
            return [rew,]
        plotter = PlayPlot(callback, 30 * 5, ["reward"])

        env = gym.make("Pong-v4")
        play(env, callback=plotter.callback)


    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:

            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """
    rendered = env.render(mode='rgb_array')
    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))
    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    pygame.font.init()
    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()
    pygame.display.set_caption('NAVI')

    info = None
    f = 0
    start = time.time()
    while running:
        f += 1
        if time.time() - start > 1:
            # print(f)
            start = time.time()
            f = 0
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), 6)
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)

            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info)

        if obs is not None:
            rendered = env.render(mode='rgb_array')
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)
        if info is not None:
            achieved_goal = info.get('achieved_goal')
            # text = achieved_goal.get('obj_type', [''])[0] + ": " + achieved_goal.get('house_number', [''])[0]
            # rel_coords = achieved_goal.get('rel_coords', [1., 1., 1., 1.])
            #display_bb(screen, rel_coords, text, video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                #print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()

class PlayPlot(object):
    def __init__(self, callback, horizon_timesteps, plot_names):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        assert plt is not None, "matplotlib backend failed, plotting will not work"

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]), c='blue')
            self.ax[i].set_xlim(xmin, xmax)
        plt.pause(0.000001)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hyrule-noop-v1', help='Define Environment')
    args = parser.parse_args()
    env = gym.make(args.env)
    # env.store_test = True
    env.test_mode = True
    play(env, zoom=4, fps=6)


if __name__ == '__main__':
    main()
