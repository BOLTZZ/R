# Machine Learning Basics:
* A Machine Learning (ML) algorithm is a type of algorithm that automatically improves itself based on experience, not by a programmer writing a better program. The algorithm gains experience by processing more and more data and then modifying itself based on the properties of the data. This is kind of how a baby learns by observing its enviroment, though the data inputs are less structured.

<strong>Types of Machine Learning Techniques:</strong>
* *Reinforcement ML* - The algorithm performs actions that will be rewarded the most, often used by game-playing AI or navigational robots.
* *Unsupervised ML* - The algorithm finds patterns in unlabeled data by clustering and identifying similarities, commonly used in recommendation systems and targeted advertising.
* *Supervised ML* - The algorithm analyzes labeled data and learns how to map input data to an output label, usually used for classification and prediction.

<strong>Neural Networks:</strong>
* A popular approach to supervised ML is the neural network. A *neural network* operates similar to how the brain is thought to work, with input flowing through many layers of "neurons" and eventually leading to an output. Deep learning uses artficial neural networks (ANN).

<img src = "https://github.com/BOLTZZ/C/blob/master/Images%20and%20Gifs/neural%20network%20outline.png" width = "500" height = "300">

Training a Network:
* Programmers don't program each neuron, but they train a neural network using a massive amount of labeled data. The training data depends on the goal of the network, if its purpose is to classify images then it could contain thousands of images with their repesctive labels. Programmers might need to modify the training set to reduce bias and not to be overfitted on the training data.
* The goal of the training phase is to determine weights for the connections between neurons that will correctly classify the training data:

<img src = "https://github.com/BOLTZZ/C/blob/master/Images%20and%20Gifs/random_weight_plane.png" width = "500" height = "300">

* The neural network starts off with all the weights set to random values, so the initial classifications are way off. But, the algorithm learns from its mistakes and eventually comes up with a set of weights that do the best job at classifying all the data:

<img src = "https://github.com/BOLTZZ/C/blob/master/Images%20and%20Gifs/finalized_weight_plane.png" width = "500" height = "300">

<strong>Accuracy:</strong>
* The accuracy of a neural network is heavily dependent on its training data (like any other ML algorithm), both the amount and diversity. Has the network seen the object from multiple angles and lighting conditions? Has it really seen all varities of that object? And, so on. Even though, ML is called "artificial intelligence", the ML algorithm is only intelligent as its training data. Programmers try to remove as much bias from training data as they can.
# Simulation Basics:
* A simulation is an abstraction of an infinitely complex natural phenomena, trying to remove details that aren't necessary or are to difficult to simulate. The level of abstraction in a simulation depends on the reason of creating it in the firstplace. Like, flight simulators for training pilots need much less abstraction than a flight simulator for gamers. 
* Simulations can be educational, entertainment, or research focused simulations.

<strong>Creating a Simulation:</strong>
* We could create a simulation for a tortoise racing a hare, based on the tortoise and hare fable.
1. We can start of simple by simulating a race between a tortoise and a hared that doesn't take naps. We can establish some initial conditions and values.
2. Now, we can add more complexity, by adding some properties that model the behavior of a lazy hare, like defining when he naps, how long the naps are, and how long he's typically awake between naps.
3. We can create a friendly Graphical User Interface (GUI) to visualize the outputs.
4. Lastly, we can add variability.
