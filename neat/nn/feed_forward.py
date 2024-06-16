from neat.graphs import feed_forward_layers
from neat.activations_jax import ActivationFunctionSet
import numpy as np
import jax
from jax import numpy as jnp
import itertools
from evojax.policy.base import PolicyNetwork


jnp_array = jax.jit(jnp.array)


class FeedForwardNetwork(object):

    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.output_size = len(outputs)
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func_idx, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        return [self.values[i] for i in self.output_nodes]

    ########## START NOT WRITTEN BY ME (but slightly modified) ##########
    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation = 'tanh' if node in config.genome_config.output_keys else ng.activation # force tanh activation on output nodes
                activation_function_idx = config.genome_config.activation_defs.name2idx[activation]
                activation_function = config.genome_config.activation_defs.get(activation)
                node_evals.append((
                    node, activation_function_idx, activation, activation_function, aggregation_function, ng.bias, ng.response, inputs
                ))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)
    ########## END NOT WRITTEN BY ME (but slightly modified) ##########
    
    @staticmethod
    def get_key2idx(input_keys, output_keys, node_keys):
        """Map nodes' keys to indices from 0 to n_nodes - 1."""
        k2i = {}
        i = 1 # index 0 is for bias
        for keys in [input_keys, output_keys, node_keys]:
            for k in keys:
                if k in k2i: # to avoid rewriting output_keys which is contained in node_keys
                    continue
                k2i[k] = i
                i += 1
        return k2i
    
    @staticmethod
    def create_params(genome, config, max_nodes=None):
        """
        Receives a genome and returns its phenotype in the form of a jnp.array
        containing all the parameters (weights & activation ids).
        Args:
            - genome (DefaultGenome)
            - config (ConfigParameter)
            - max_nodes (int): Maximum number of nodes. Used to pad the matrices to batch the parameters.
        Returns:
            - params (jnp.array): shape (n_nodes, n_nodes + 1). weights & activation ids
            - net (FeedForwardNetwork)
        """
        net = FeedForwardNetwork.create(genome, config)

        # Map a node's key to its future index in the parameters array.
        k2i = net.get_key2idx(
            net.input_nodes, net.output_nodes,
            [x[0] for x in net.node_evals], # we must preserve the order from net.node_evals because of nodes dependencies
        )

        if max_nodes is None: # for inference time when there is no batching
            max_nodes = 1 + len(k2i)
        W = np.zeros((max_nodes, max_nodes)) # W[i, j] is the weight from node i to j.
        A = np.zeros((max_nodes,)) # A[i] is the activation function of node i.

        for node, act_func_idx, act, act_func, agg_func, bias, response, links in net.node_evals:
            ni = k2i[node]
            for i, w in links:
                W[k2i[i], ni] = w
            W[0, ni] = bias
            A[ni] = act_func_idx
        return jnp_array(np.c_[W, A]), net


act_lut = ActivationFunctionSet().lut # activation functions lookup table

def forward(inputs, output_size, W, A):
    """
    Forward pass of neural net defined by W (weights) and A (activation ids) on data `inputs`.
    Args:
        - inputs (jnp.array)
        - output_size (int)
        - W (jnp.array): connections' weights
        - A (jnp.array): nodes' activation ids
    Returns:
        - outputs (jnp.array)
    """
    input_size = inputs.shape[0]
    max_nodes = W.shape[0]
    n1 = 1 + input_size # points to the 1st entry of the outputs
    n2 = n1 + output_size # points to the 1st entry of the node activations

    # node activations which will gradually be populated through the forward pass
    node_acts = jnp.r_[1, inputs, jnp.zeros(max_nodes - n1)] # first entry is bias

    for i in itertools.chain(range(n2, max_nodes), range(n1, n2)):
        x = jnp.dot(node_acts, W[:, i]) # multiply input activations with weights
        x = jax.lax.switch(A[i].astype(jnp.int8), act_lut, x) # apply activation function
        node_acts = node_acts.at[i].set(x)
    return node_acts[n1:n2]

batch_forward = jax.jit(
    jax.vmap(
        forward,
        in_axes=(0, None, 0, 0),
        out_axes=0,
    ),
    static_argnames=['output_size'],
)

class FeedForwardPolicy(PolicyNetwork):

    def __init__(self, output_size):
        self.output_size = output_size
    
    def get_actions(self, t_states, params, p_states):
        inputs = t_states.obs # shape (pop_size, input_size)
        W = params[:, :, :-1] # shape (pop_size, n_nodes, n_nodes)
        A = params[:, :, -1] # shape (pop_size, n_nodes)
        outputs = batch_forward(inputs, self.output_size, W, A)
        return outputs, p_states