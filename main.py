import optuna
from dqn import DQN



def tune_hyperparameters():

    def objective(trial):
        gamma = trial.suggest_float('gamma', 0.8, 0.99)
        epsilon_start = trial.suggest_float('epsilon_start', 0.6, 1.0)
        epsilon_min = trial.suggest_float('epsilon_min', 0.01, 0.1)
        epsilon_decay = trial.suggest_float('epsilon_decay', 0.02, 0.2)
        C = trial.suggest_int('C', 10, 1000)
        buffer_batch_size = trial.suggest_int('buffer_batch_size', 50, 500)

        agent = DQN(
            episode_count=1000,
            timestep_count=100,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            C=C,
            buffer_batch_size=buffer_batch_size,
        )

        best_reward = agent.train()

        return best_reward

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best reward: {study.best_value}")


if __name__ == "__main__":
        
    dqn = DQN(
        episode_count=400,
        timestep_count=100,
        gamma=0.9,
        epsilon_start=0.9,
        epsilon_min=0.01,
        epsilon_decay=0.02,
        C=50,
        buffer_batch_size=100,
    )

    # dqn.train()
    tune_hyperparameters()