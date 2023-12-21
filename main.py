import gymnasium as gymnasium  

env = gymnasium.make('Acrobot-v1', render_mode="human")

start_values = env.reset()
print('Start values:', start_values)

actions = [0,1,2]
action = 0

for i in range(10):

    current_state = env.step(action)
    # print("Current state:", current_state)

    (current_state, reward, terminated, truncated, info) = current_state
    print("Current state:", current_state)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)
    print()

    env.render()

env.close()
