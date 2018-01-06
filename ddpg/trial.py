import tensorflow as tf
import numpy as np
from logger import Logger
from memory import Memory
from envWrapper import NoGoalCMC, NoGoalHalfCheetah
from agents.actor import ActorNetwork
from agents.critic import CriticNetwork
from agents.stopfirstrewardAgent import stop_first_reward_agent
from agents.myddpgAgent import DDPG_agent
from agents.fbddpgAgent import FB_DDPG_agent
from agents.oflddpgAgent import OFL_DDPG_agent
from noise import OrnsteinUhlenbeckActionNoise, NoNoise


def trial(config):
    # Get the environment and extract the number of actions.
    env = config.env

    if config.Name==None:
        results_path = './' + config.results_root_name + '/{}/{}/'.format(config.tau, config.trial)
    else:
        results_path = './' + config.results_root_name + '/{}/{}/'.format(config.name, config.trial)
    # logger_step = Logger(dir=results_path,format_strs=['log','json', 'tensorboard'])
    # logger_episode = Logger(dir=results_path, format_strs=['log','stdout', 'json', 'tensorboard'])
    if (config.save_step_stats):
        logger_step = Logger(dir=results_path+'/log_steps', format_strs=['json', 'tensorboard'])
        logger_episode = Logger(dir=results_path+'/log_episodes', format_strs=['json', 'tensorboard'])
    else:
        logger_step = Logger(dir=results_path + '/log_steps', format_strs=['json'])
        logger_episode = Logger(dir=results_path + '/log_episodes', format_strs=['json'])

    if config.type=="cmc":
        env_wrapper = NoGoalCMC()
    elif config.type=="halfcheetah":
        env_wrapper = NoGoalHalfCheetah()
    else:
        print("env type unknow:", config.type)

    state_dim = env_wrapper.state_shape[0]
    action_dim = env_wrapper.action_shape[0]
    action_bound = env.action_space.high
    # Ensure action bound is symmetric
    #assert (env.action_space.high == -env.action_space.low)

    memory = Memory(env_wrapper, with_reward=True, limit=int(1e6))
    if config.study == "offline":
        memory.load_from_ManceronBuffer(file=config.memory_file)
    elif config.study == "from_cedric" or config.study=="from_cedric_ofl":
        memory.load_from_Cedric(config.buffer_name)
        nb_success = memory.count_success_from_Cedric(config.buffer_name)
        print(nb_success)

    # Noise
    if config.study == "offline" or config.study=="from_cedric_ofl" or config.study=="from_cedric":
        actor_noise = NoNoise()
    else:
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma = config.noise_factor)

    with tf.Session() as sess:


        critic = CriticNetwork(sess,
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

        if config.study=="first":
            agent = stop_first_reward_agent(sess,
                            actor,
                            actor_noise,
                            critic,
                            env,
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
                            config.save_step_stats,
                            config.averaging)
            nb_steps = agent.run()
            return nb_steps

        elif config.study=="offline" or config.study=="from_cedric_ofl":
            update_actor = ActorNetwork(sess,
                                        state_dim,
                                        action_dim,
                                        action_bound,
                                        config.tau,
                                        config.actor_lr)

            # update_actor.save_weights('good_actor.p')
            update_actor.load_weights('actors/good_actor.save')
            update_actor.load_target_weights('actors/good_actor.save')
            # portrait_actor(update_actor.target_model, env, save_figure=True, figure_file="./img/update_actor.png")

            agent = OFL_DDPG_agent(update_actor,
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
                                   config.save_step_stats,
                                   config.averaging)
        elif config.study == "standard" or config.study=="from_cedric":
            if config.frozen:
                agent = FB_DDPG_agent(sess,
                                   actor,
                                   actor_noise,
                                   critic,
                                   env,
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
                                   config.save_step_stats,
                                   config.averaging)
            else:
                agent = DDPG_agent(sess,
                               actor,
                               actor_noise,
                               critic,
                               env,
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
                               config.save_step_stats,
                               config.averaging)
        else:
            print("config.study not known",config.study)

        agent.run()
    
