# Slime Volleyball AI

In this project, I trained an agent to play Slime Volleyball using the [NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) algorithm (NeuralEvolution of Augmenting Topologies).

<p align="center">
Left: Baseline - Right: NEAT agent n°1 </br>
<img width=600 src="images/agent1_play.gif">
</p>

<p align="center">
Neural net of NEAT agent n°1 </br>
Avg score: 0.73 </br>
<img width=600 src="images/agent1_graph.png">
</p>

NEAT is an evolutionay algorithm which evolves a neural net starting from a simple one, and progressively adding or removing nodes and connections and changing the weights to maximize the fitness function, which is in this case the score obtained against a baseline after 3000 time steps of game play.

After roughly 1000 generations, NEAT tends to evolve a simple neural net with few nodes. The neural net tends to ignore the opponent's information, making the resulting agent robust to different kinds of opponent.

The best agent I could evolve gets an average score of 0.77 ± 0.92 against the baseline (1024 trials and 3000 time steps per trial).

<p align="center">
Left: Baseline - Right: NEAT agent n°2 </br>
<img width=600 src="images/agent2_play.gif">
</p>

<p align="center">
Neural net of NEAT agent n°2 </br>
Avg score: 0.77 </br>
<img width=600 src="images/agent2_graph.png">
</p>