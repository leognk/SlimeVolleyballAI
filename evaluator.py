import jax
from jax import numpy as jnp

from slimevolley import SlimeVolley
from obs_norm import ObsNormalizer
from sim_mgr import SimManager
import neat


jnp_array = jax.jit(jnp.array)

class Evaluator:

    def __init__(self, pop_size, output_size, max_steps, n_repeats, seed):
        self.policy = neat.nn.FeedForwardPolicy(output_size)
        self.train_task = SlimeVolley(test=False, max_steps=max_steps)
        self.test_task = SlimeVolley(test=True, max_steps=max_steps)
        self.obs_normalizer = ObsNormalizer(
            obs_shape=self.train_task.obs_shape,
            dummy=True,
        )
        self.sm = SimManager(
            n_repeats=n_repeats,
            test_n_repeats=1,
            pop_size=pop_size,
            n_evaluations=n_repeats,
            policy_net=self.policy,
            train_vec_task=self.train_task,
            valid_vec_task=self.test_task,
            seed=seed,
            obs_normalizer=self.obs_normalizer,
            use_for_loop=False,
        )
    
    def get_nb_nodes(self, genome, config):
        # bias + nb inputs + nb outputs + nb hidden nodes
        return 1 + config.genome_config.num_inputs + len(genome.nodes)

    def get_scores(self, genomes, config, reduce=True):
        """Create neural nets' parameters from their genomes and compute their respective score."""
        max_nodes = max(self.get_nb_nodes(genome, config) for _, genome in genomes)
        batch_params = []
        for _, genome in genomes:
            batch_params.append(
                neat.nn.FeedForwardNetwork.create_params(genome, config, max_nodes)[0])
        scores, _ = self.sm.eval_params(params=jnp_array(batch_params), test=False, reduce=reduce)
        return scores

    def eval_genomes(self, genomes, config):
        """Compute and update genomes' fitness attribute."""
        scores = self.get_scores(genomes, config)
        for (_, genome), score in zip(genomes, scores):
            genome.fitness = score