import os
from pathlib import Path
import time
from datetime import timedelta

import numpy as np
import jax

from slimevolley import SlimeVolley
import neat
import visualize
from evaluator import *


def run(config_file, log_dir):
    np.random.seed(0)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    output_size = config.genome_config.num_outputs
    max_steps = 3000

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(250, dir=log_dir))

    evaluator = Evaluator(
        config.pop_size, output_size, max_steps, n_repeats=128, seed=0
    )

    winner = p.run(evaluator.eval_genomes, n=1500) # n is the number of generations

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, True, filename="winner-net.gv", dir=log_dir)
    visualize.draw_net(config, winner, True, filename="winner-net-pruned.gv", dir=log_dir, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True, dir=log_dir)
    visualize.plot_species(stats, view=True, dir=log_dir)

    # Test best agent
    n_repeats_test = 1000
    evaluator = Evaluator(1, output_size, max_steps, n_repeats=n_repeats_test, seed=0)
    scores = evaluator.get_scores([(None, winner)], config, reduce=False)
    scores = np.array(scores)[0]

    print(
        f"\nBest agent ({n_repeats_test} trials): "
        f"mean={scores.mean():.3f} | std={scores.std():.3f} | min={scores.min():.3f} | max={scores.max():.3f}"
    )
    
    # Visualize best agent's play
    task_reset_fn = jax.jit(evaluator.test_task.reset)
    policy_reset_fn = jax.jit(evaluator.policy.reset)
    step_fn = jax.jit(evaluator.test_task.step)
    action_fn = jax.jit(evaluator.policy.get_actions)
    best_params, _ = neat.nn.FeedForwardNetwork.create_params(winner, config)
    best_params = jnp_array([best_params])
    key = jax.random.PRNGKey(0)[None, :]
    task_state = task_reset_fn(key)

    policy_state = policy_reset_fn(task_state)
    screens = []
    for _ in range(max_steps):
        action, policy_state = action_fn(task_state, best_params, policy_state)
        task_state, reward, done = step_fn(task_state, action)
        screens.append(SlimeVolley.render(task_state))

    gif_file = os.path.join(log_dir, 'best_agent_play.gif')
    screens[0].save(gif_file, save_all=True, append_images=screens[1:], duration=20, loop=0)
    print(f'\nGIF saved to {gif_file}.')


if __name__ == '__main__':
    start_time = time.time()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'configs', 'config1')
    log_dir = os.path.join(local_dir, 'log', 'log1')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    run(config_path, log_dir)
    dt = time.time() - start_time
    print(f"\nProgram duration: {timedelta(seconds=round(dt))}")