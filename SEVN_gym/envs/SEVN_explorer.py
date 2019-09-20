from __future__ import print_function, division
from SEVN_gym.envs.SEVN_base import SEVNBase
from SEVN_gym.envs import utils, wrappers
import time

class SEVNExplorer(SEVNBase):
    def __init__(self, obs_shape=(4, 84, 84), use_image_obs=True,
                 use_gps_obs=False, use_visible_text_obs=True,
                 use_full=False, reward_type=None):
        super(SEVNExplorer, self).__init__(obs_shape, use_image_obs, False,
                                           use_visible_text_obs, use_full,
                                           reward_type)
        self.max_num_steps = 300
        self.seen_house_nums = []

    def step(self, a):
        start = time.time()
        done = False

        reward = 0.0
        self.num_steps_taken += 1
        action = self._action_set(a)
        image, x, w = self._get_image()
        visible_text = self._get_visible_text(x, w)

        if self.num_steps_taken >= self.max_num_steps and done is False:
            done = True
        elif action == self.Actions.FORWARD:
            self.transition()
        else:
            self.turn(action)
        reward = self.compute_reward(x, {'visible_text': visible_text}, done)
        obs = {'image': image, 'visible_text': visible_text}
        obs = wrappers.wrap_obs(obs, self.use_gps_obs,
                                self.use_visible_text_obs,
                                self.use_image_obs, False, self.num_streets)

        info = {}
        if done:
            self.needs_reset = True
        print(f'step: {time.time() - start}')

        return obs, reward, done, info

    def reset(self):
        self.needs_reset = False
        self.num_steps_taken = 0
        self.seen_house_nums = []
        image, x, w = self._get_image()
        obs = {'image': image, 'visible_text': self._get_visible_text(x, w)}
        obs = wrappers.wrap_obs(obs, self.use_gps_obs,
                                self.use_visible_text_obs,
                                self.use_image_obs, False, self.num_streets)
        return obs

    def compute_reward(self, x, info, done):
        '''
        Compute the step reward. This externalizes the reward function and
        makes it dependent on an a desired goal and the one that was achieved.
        If you wish to include additional rewards that are independent of the
        goal, you can include the necessary values to derive it in info and
        compute it accordingly.

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to
            attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal
            w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'],
                                                    ob['goal'], info)
        '''
        reward = 0
        house_numbers = info['visible_text']['house_numbers']
        house_numbers = utils.convert_house_vec_to_ints(house_numbers)
        for num in house_numbers:
            if num not in self.seen_house_nums:
                reward += 1
                self.seen_house_nums.append(num)
        return reward
