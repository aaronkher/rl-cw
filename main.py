import gymnasium as gymnasium  

env = gymnasium.make('Acrobot-v1', render_mode="human")

start_values = env.reset()
print('Start values:', start_values)


# actions = [0,1,2]
action = 0

for i in range(10):
    values = env.step(action)

    print("Values:")
    print(values)

    env.render()

env.close()
