import neat
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
from evaluator import *
import visualize


log_dir = os.path.join("log", "log2")
ckpt = 1666

seed = 0
n_repeats = 1024
n_repeats_batch = 128
n_plays = 1


save_dir = os.path.join(log_dir, f"viz_checkpoint-{ckpt}")
Path(save_dir).mkdir(parents=True, exist_ok=True)

ckpt_file = f"neat-checkpoint-{ckpt}"
ckpt_path = os.path.join(log_dir, ckpt_file)
p = neat.Checkpointer.restore_checkpoint(ckpt_path)

species = p.species.species
config = p.config
output_size = config.genome_config.num_outputs
max_steps = 3000

evaluator = Evaluator(config.pop_size, output_size, max_steps, n_repeats=n_repeats_batch, seed=seed)

task_reset_fn = jax.jit(evaluator.test_task.reset)
policy_reset_fn = jax.jit(evaluator.policy.reset)
step_fn = jax.jit(evaluator.test_task.step)
action_fn = jax.jit(evaluator.policy.get_actions)
key = jax.random.key(seed)

# Calculate each species' members scores
n_batches = n_repeats // n_repeats_batch
scores = {} # species -> list of size (n_members, n_repeats)
genomes = {} # species -> list of size (n_members)
for species_key, this_species in species.items():
    this_genomes = list(this_species.members.items())
    evaluator.sm._pop_size = len(this_genomes)
    species_scores = []
    for _ in range(n_batches):
        batch_scores = evaluator.get_scores(this_genomes, config, reduce=False)
        species_scores.append(batch_scores)
    species_scores = np.concatenate(species_scores, axis=1)
    scores[species_key] = species_scores
    genomes[species_key] = this_genomes

score_file = os.path.join(save_dir, "scores.txt")
with open(score_file, 'w'): pass

# Iterate over the species
for species_key in species.keys():

    species_scores = scores[species_key]
    avg_scores = np.mean(species_scores, axis=-1)
    best_genome_idx = np.argmax(avg_scores)
    best_scores = species_scores[best_genome_idx]

    # Save and print the species' best agent's score
    score_str = (
        f"\n\nSpecies {species_key}"
        f"\nBest agent ({n_repeats} trials): "
        f"mean={best_scores.mean():.3f} | std={best_scores.std():.3f}"
        f" | min={best_scores.min():.3f} | max={best_scores.max():.3f}"
    )
    with open(score_file, 'a') as f:
        f.write(score_str)
    print(score_str)

    best_genome = genomes[species_key][best_genome_idx][1]
    best_params, best_net = neat.nn.FeedForwardNetwork.create_params(best_genome, config)
    best_params = jnp_array([best_params])

    # Draw and save the genome's neural net
    activations = {node[0]: node[2] for node in best_net.node_evals}
    node_names = {
        **activations,

        # inputs
         -1: "x",
         -2: "y",
         -3: "vx",
         -4: "vy",
         -5: "ball\nx",
         -6: "ball\ny",
         -7: "ball\nvx",
         -8: "ball\nvy",
         -9: "opp\nx",
        -10: "opp\ny",
        -11: "opp\nvx",
        -12: "opp\nvy",

        # outputs
        0: f"forward\n{activations[0]}",
        1: f"backward\n{activations[1]}",
        2: f"jump\n{activations[2]}",
    }
    visualize.draw_net(
        config, best_genome, True,
        filename=f"species_{species_key}_net", dir=save_dir,
        node_names=node_names,
    )
    visualize.draw_net(
        config, best_genome, True,
        filename=f"species_{species_key}_net-pruned", dir=save_dir,
        node_names=node_names, prune_unused=True,
    )

    # Perform several plays with best agent and save in GIF files
    for play_idx in tqdm(range(n_plays)):
        key, subkey = jax.random.split(key=key)
        task_state = task_reset_fn(subkey.reshape(1))
        policy_state = policy_reset_fn(task_state)
        screens = []
        score = 0
        for _ in range(max_steps):
            action, policy_state = action_fn(task_state, best_params, policy_state)
            task_state, reward, done = step_fn(task_state, action)
            screens.append(SlimeVolley.render(task_state))
            score += reward.item()
        species_dir = os.path.join(save_dir, f"species_{species_key}")
        Path(species_dir).mkdir(parents=True, exist_ok=True)
        gif_file = os.path.join(species_dir, f'play-{play_idx + 1}_score={score}.gif')
        screens[0].save(gif_file, save_all=True, append_images=screens[1:1990:2], duration=40, loop=0)