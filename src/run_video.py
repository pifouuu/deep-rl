from keras.models import load_model
import os
from gym.envs.registration import make
from gym.monitoring import VideoRecorder
import numpy as np
from ddpg.util import load

log_dir = '../log/cluster1804/ReacherGoal-v0_fixed_goal_no_4_1_10_0.0001_10_0.3_1_1_sparse_0.02_uni_1000_100/20180417182320_658384/'
model = load_model(os.path.join(log_dir, 'saves', 'target_actor_model.h5'))
test_env = make('ReacherGoal-v0')

if test_env.spec._goal_wrapper_entry_point is not None:
    wrapper_cls = load(test_env.spec._goal_wrapper_entry_point)
    test_env = wrapper_cls(test_env, 'sparse', 0.02)




vid_dir = os.path.join(log_dir, 'videos')
os.makedirs(vid_dir, exist_ok=True)
base_path = os.path.join(vid_dir, 'video_init')
rec = VideoRecorder(test_env, base_path=base_path)
test_env.rec = rec

for _ in range(20):
    test_env.goal = test_env.sample_test_goal()

    print(test_env.goal)

    state = test_env.reset()
    reward_sum = 0
    for k in range(50):
        test_env.render(mode='human')
        action = model.predict(np.reshape(state, (1, test_env.state_dim[0])))
        action = np.clip(action, test_env.action_space.low, test_env.action_space.high)
        state, reward, terminal, info = test_env.step(action[0])
        reward_sum += reward
        if terminal:
            print(reward_sum)
            break
