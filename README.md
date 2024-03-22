# Quantum Artificial intelligence Demo

Quantum artificial intelligence can be summarized with the three equations in the image. With the exception of the metric, those equations should look familiar in Euclidean geometry as a dot product and a cosine similarity. In quantum mechanics, those equations compute the transition probability and since they are normalized they easily deal with the issue of curse of dimensionality that occurs in higher dimensional Euclidean space.

The demo will be started by loading basic pandas, numpy and basic math functions libraries.
Since We will be working with the familiar Digit Recognizer dataset for this demo, we will load the dataset and prepare the training and testing dataset.

Next, we will define the metric function and generate a metric for the dataset. Then, we will compute the transition probability also known as similarity in Euclidian geometry. From the similarity matrix, we can easily lookup digits with the highest similarity probability.

Finally, we will define a non-Euclidean scoring metric to evaluate our quantum artificial intelligence model. That's it for the demo! Let me know what you built with it and how quantum artificial intelligence helped you deal with the curse of dimensionality.
