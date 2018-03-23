from keras.models import load_model
import os
from gym.envs.registration import make
from gym.monitoring import VideoRecorder
import numpy as np
from ddpg.util import load

log_dir = '../log/local/CMCPos-v0/20180323131658_047198/'
model = load_model(os.path.join(log_dir, 'saves', 'actor_model.h5'))
test_env = make('CMCPos-v0')
if test_env.spec._goal_wrapper_entry_point is not None:
    wrapper_cls = load(test_env.spec._goal_wrapper_entry_point)
    test_env = wrapper_cls(test_env)

vid_dir = os.path.join(log_dir, 'videos')
os.makedirs(vid_dir, exist_ok=True)
base_path = os.path.join(vid_dir, 'video_random')
rec = VideoRecorder(test_env, base_path=base_path)
test_env.rec = rec

test_env.set_goal_reachable()
state = test_env.reset()
for k in range(1000):
    action = model.predict(np.reshape(state, (1, test_env.state_dim[0])))
    action = np.clip(action, test_env.action_space.low, test_env.action_space.high)
    state, reward, terminal, info = test_env.step(action[0])
    terminal = terminal or info['past_limit']
    if terminal:
        break
