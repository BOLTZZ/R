# Machine Learning Basics:
* A Machine Learning (ML) algorithm is a type of algorithm that automatically improves itself based on experience, not by a programmer writing a better program. The algorithm gains experience by processing more and more data and then modifying itself based on the properties of the data. This is kind of how a baby learns by observing its enviroment, though the data inputs are less structured.

<strong>Types of Machine Learning Techniques:</strong>
* *Reinforcement ML* - The algorithm performs actions that will be rewarded the most, often used by game-playing AI or navigational robots.
* *Unsupervised ML* - The algorithm finds patterns in unlabeled data by clustering and identifying similarities, commonly used in recommendation systems and targeted advertising.
* *Supervised ML* - The algorithm analyzes labeled data and learns how to map input data to an output label, usually used for classification and prediction.

<strong>Neural Networks:</strong>
* A popular approach to supervised ML is the neural network. A *neural network* operates similar to how the brain is thought to work, with input flowing through many layers of "neurons" and eventually leading to an output. Deep learning uses artficial neural networks (ANN).

<img src = "cdn.kastatic.org/ka-perseus-images/5209c097f94ad035f1201d56428e52e0d7811481.svg" width = "600" height = "300">

Training a Network:
* Programmers don't program each neuron, but they train a neural network using a massive amount of labeled data. The training data depends on the goal of the network, if its purpose is to classify images then it could contain thousands of images with their repesctive labels. Programmers might need to modify the training set to reduce bias and not to be overfitted on the training data.
