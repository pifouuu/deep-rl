import tensorflow as tf
import numpy as np
from logger import Logger
from memory import Memory
from envWrapper import NoGoal
from actor import ActorNetwork
from HLcritic import HuberLossCriticNetwork
from ofpddpgAgent import OFPDDPG_agent
from noise import OrnsteinUhlenbeckActionNoise, NoNoise
from plot import portrait_actor

def perf_study_ofp(delta_clip, num, config):
    # Get the environment and extract the number of actions.
    env = config.env

    results_path = './perf_ofp/{}/{}/'.format(delta_clip, num)
    # logger_step = Logger(dir=results_path,format_strs=['log','json', 'tensorboard'])
    # logger_episode = Logger(dir=results_path, format_strs=['log','stdout', 'json', 'tensorboard'])
    if (config.save_step_stats):
        logger_step = Logger(dir=results_path+'/log_steps', format_strs=['json', 'tensorboard'])
        logger_episode = Logger(dir=results_path+'/log_episodes', format_strs=['json', 'tensorboard'])
    else:
        logger_step = Logger(dir=results_path + '/log_steps', format_strs=['json'])
        logger_episode = Logger(dir=results_path + '/log_episodes', format_strs=['json'])

    env_wrapper = NoGoal()

    state_dim = env_wrapper.state_shape[0]
    action_dim = env_wrapper.action_shape[0]
    action_bound = env.action_space.high
    # Ensure action bound is symmetric
    assert (env.action_space.high == -env.action_space.low)

    memory = Memory(env_wrapper, with_reward=True, limit=int(1e6))
    memory.load_from_ManceronBuffer(file=config.memory_file)

    # Noise
    # actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    actor_noise = NoNoise()

    with tf.Session() as sess:
        if not config.random_seed:
            np.random.seed(config.seed)
            tf.set_random_seed(config.seed)
            env.seed(config.seed)

        critic = HuberLossCriticNetwork(delta_clip,
                                        sess,
                                        state_dim,
                                        action_dim,
                                        config.gamma,
                                        config.tau,
                                        config.critic_lr)
        actor = ActorNetwork(sess,
                             state_dim,
                             action_dim,
                             action_bound,
                             config.tau,
                             config.actor_lr)

        update_actor = ActorNetwork(sess,
                                    state_dim,
                                    action_dim,
                                    action_bound,
                                    config.tau,
                                    config.actor_lr)

        # update_actor.save_weights('good_actor.p')
        update_actor.load_weights('actors/good_actor.save')
        update_actor.load_target_weights('actors/good_actor.save')
        #portrait_actor(update_actor.target_model, env, save_figure=True, figure_file="./img/update_actor.png")

        agent = OFPDDPG_agent(update_actor,
                              sess,
                              actor,
                              actor_noise,
                              critic,
                              env,
                              env_wrapper,
                              memory,
                              logger_step,
                              logger_episode,
                              config.batch_size,
                              config.eval_episodes,
                              config.max_episode_steps,
                              config.max_steps,
                              config.eval_freq,
                              config.save_step_stats)

        agent.run()
