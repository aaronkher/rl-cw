import sys

from ddpg.ddpg import DDPG
from dqn.dqn import DQN
from environment.gymnasium import CartpoleEnv, MountainCarEnv, PendulumEnv
from environment.trackmania import ControllerTrackmania, KeyboardTrackmania
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("checkpoint_id", type=str, help="The checkpoint id to load from", nargs="?")
    parser.add_argument("--epsilon-start", type=float, help="The epsilon start value")
    parser.add_argument("--dqn", action="store_true")
    parser.add_argument("--ddpg", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint_id: str | None = args.checkpoint_id
    epsilon_start: float | None = args.epsilon_start

    episode_count = 10**100
    timestep_count = 10**100
    if args.dqn:
        env = KeyboardTrackmania()
        create_agent = lambda chk=checkpoint_id, eps=episode_count: DQN(
            environment=env,
            episode_count=eps,
            timestep_count=timestep_count,
            gamma=1.0,
            epsilon_start=args.epsilon_start or 0.3,
            epsilon_min=0.01,
            epsilon_decay=0.01,
            buffer_batch_size=256,
            checkpoint_id=chk,
            # vanilla=True,
        )
    elif args.ddpg:
        env = ControllerTrackmania()
        create_agent = lambda chk=checkpoint_id, eps=episode_count: DDPG(
            environment=env,
            episode_count=eps,
            gamma=1.0,
            buffer_batch_size=256,
            target_network_learning_rate=0.005,
            checkpoint_id=chk,
        )
    else:
        print("Either --dqn or --ddpg must be specified!")
        sys.exit(1)

    if checkpoint_id is not None:
        print(f"Loading agent from checkpoint: {checkpoint_id}\n")
        agent = create_agent()
        agent.train()
        sys.exit(0)
    else:
        print("No checkpoint id provided, training new agent\n")
        high_score = float("inf")
        best_agent: DQN | DDPG | None = None

        for i in range(9):
            print(f"Training agent: {i+1}\n")
            agent = create_agent(eps=500000)
            agent.train()

            if agent.high_score < high_score:
                print(f"Starting agent upgraded, new highest score: {agent.high_score}\n")
                best_agent = agent
                high_score = agent.high_score

        assert best_agent is not None
        print("Training best agent \n")
        best_agent_continued = create_agent(chk=best_agent.latest_checkpoint)
        best_agent_continued.train()
