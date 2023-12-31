from .pytorch_example import DDPG
from environment import Environment
from data_helper import plot_episode_data, EpisodeData, create_figure
import gymnasium

env = Environment("Pendulum-v1")

DDPG( gamma=0.99, 
    tau=0.005, 
    hidden_size=128, 
    num_inputs=env.observation_space_length, 
    action_space=env.action_count, 
    checkpoint_dir=None
    )

episode_count = 1000
timestep_count = 1000
sigma = 0.5

episodes = []
mean_rewards = []
create_figure()
try:
    # timestep_C_count = 0
    for episode in range(episode_count):
        print(f"Episode: {episode}")
        state = env.reset()

        timestep = 0
        reward_sum = 0
        won = False

        for timestep in range(timestep_count):
            sigma = sigma * (0.999) # decaying exploration noise
            # state = env.current_state  # S_t

            # action = self.get_action(state)  # A_t
            action = DDPG.calc_action(state, sigma)
            action_result = self.execute_action(action)

            reward_sum += action_result.reward

            # print(
            #     f"Episode {episode} Timestep {timestep} | Action {action}, Reward {action_result.reward:.0f}, Total Reward {reward_sum:.0f}"
            # )

            experience = Experience(
                action_result.old_state,
                action_result.new_state,
                action,
                action_result.reward,
                action_result.terminal and not action_result.won,
            )
            self.replay_buffer.add_experience(experience)

            if self.replay_buffer.size() > self.buffer_batch_size:
                replay_batch = self.replay_buffer.get_batch(
                    self.buffer_batch_size
                )
                # TODO td_target calculation isn't the same as DQN because we're using a continuous action space
                td_targets = self.compute_td_targets_batch(replay_batch) # y (target network)

                # Pass back target networks values to update
                # Gradient descent on critic
                self.backprop_critic(replay_batch, td_targets)

                # Gradient ascent on actor
                self.backprop_actor(replay_batch, self.critic_network)

                self.update_target_networks()

            # timestep_C_count += 1
            # if timestep_C_count == self.C:
            #     self.update_target_network()
            #     timestep_C_count = 0

            # process termination
            if action_result.terminal:
                won = action_result.won
                won_str = "won" if won else "lost"
                print(
                    f"Episode {episode+1} ended ({won_str}) after {timestep+1} timestaps"
                    f" with total reward {reward_sum:.2f}"
                )
                break

        if (episode % 10 == 0):
            # mean_rewards.append(np.mean([episode.reward for episode in episodes[-10:]]))
            plot_episode_data(episodes) # comment out if you don't want live plot updates
        episodes.append(EpisodeData(episode, reward_sum, timestep, won))
        self.decay += 1
        # self.decay_epsilon(episode)
        # print(f"Episode {episode} finished with total reward {reward_sum}")

except KeyboardInterrupt:
    pass

plot_episode_data(episodes)