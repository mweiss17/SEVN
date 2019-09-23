from __future__ import print_function, division

from SEVN_gym.envs import utils, wrappers
from SEVN_gym.envs.SEVN_base import SEVNBase


ACTION_MEANING = {
    0: 'LEFT_BIG',
    1: 'LEFT_SMALL',
    2: 'FORWARD',
    3: 'RIGHT_SMALL',
    4: 'RIGHT_BIG',
    5: 'DONE',
    6: 'NOOP',
    7: 'READ'
}


class SEVNCostlyText(SEVNBase):
    def __init__(self, obs_shape=(4, 84, 84), use_image_obs=True,
                 use_gps_obs=True, use_visible_text_obs=False,
                 use_full=False, reward_type=None):
        super(SEVNCostlyText, self).__init__(obs_shape, use_image_obs,
                                             use_gps_obs, use_visible_text_obs,
                                             use_full, reward_type)

    def step(self, a):
        done = False
        was_successful_trajectory = False
        oracle = False

        reward = 0.0
        self.num_steps_taken += 1
        action = self._action_set(a)
        if oracle:
            action = next(iter(self.shortest_path_length()), None)
        image, x, w = self._get_image()
        visible_text = self._get_visible_text(x, w)
        self.use_visible_text_obs = False
        if action == self.actions.READ:
            self.use_visible_text_obs = True

        try:
            if self.is_successful_trajectory(x):
                done = True
                was_successful_trajectory = True
            elif self.num_steps_taken >= self.max_num_steps and done is False:
                done = True
            elif action == self.Actions.FORWARD:
                self.transition()
            else:
                self.turn(action)
        except Exception:
            self.is_successful_trajectory(x)

        reward = self.compute_reward(x, {}, done, action)

        self.agent_gps = utils.sample_gps(self.meta_df.loc[self.agent_loc],
                                          self.x_scale, self.y_scale)
        rel_gps = [self.target_gps[0] - self.agent_gps[0],
                   self.target_gps[1] - self.agent_gps[1],
                   self.target_gps[0], self.target_gps[1]]
        obs = {'image': image, 'mission': self.goal_address,
               'rel_gps': rel_gps, 'visible_text': visible_text}
        obs = wrappers.wrap_obs(obs, self.use_gps_obs,
                                self.use_visible_text_obs, self.use_image_obs,
                                True, self.num_streets)

        info = {}
        if done:
            info['was_successful_trajectory'] = was_successful_trajectory
            self.needs_reset = True

        return obs, reward, done, info

    def compute_reward(self, x, info, done, action):
        cur_spl = len(self.shortest_path_length())
        if done and self.is_successful_trajectory(x):
            reward = 2.0
        elif done and not self.is_successful_trajectory(x):
            reward = -2.0
        elif self.prev_spl - cur_spl > 0:
            reward = 1
        elif self.prev_spl - cur_spl <= 0:
            reward = -1

        if self.reward_type == 'Sparse' and reward != 2.0 and reward != -2.0:
            reward = 0
        self.prev_spl = cur_spl

        if action == self.actions.READ:
            reward -= 0.5

        return reward
