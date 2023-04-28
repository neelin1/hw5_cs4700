import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** BEGIN YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(), x)
        "*** END YOUR CODE HERE ***"

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** BEGIN YOUR CODE HERE ***"
        if (nn.as_scalar(self.run(x)) >= 0):
            return 1
        else:
            return -1
        "*** END YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        Assume a learning rate of 1.
        """
        "*** BEGIN YOUR CODE HERE ***"
        percentAccurate = 0.0
        while (percentAccurate < 1.0):
            num = 0.0
            numAccurate = 0.0
            for x, y in dataset.iterate_once(1):
                num += 1
                if (self.get_prediction(x) != nn.as_scalar(y)):
                    # self.get_weights().update(x, nn.as_scalar(y))
                    self.w.update(nn.Constant(nn.as_scalar(y)*x.data), 1)
                else:
                    numAccurate += 1
            percentAccurate = numAccurate/num
        "*** END YOUR CODE HERE ***"


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** BEGIN YOUR CODE HERE ***"
        self.batchSize = 1
        self.learningRate = -0.01
        self.lossThreshold = 0.01
        self.w1 = nn.Parameter(1, 100)
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 200)
        self.b2 = nn.Parameter(1, 200)
        self.w3 = nn.Parameter(200, 1)
        self.b3 = nn.Parameter(1, 1)
        "*** END YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** BEGIN YOUR CODE HERE ***"
        l1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        l2 = nn.ReLU(nn.AddBias(nn.Linear(l1, self.w2), self.b2))
        # l3 = nn.ReLU(nn.AddBias(nn.Linear(l2, self.w3), self.b3))
        l3 = nn.AddBias(nn.Linear(l2, self.w3), self.b3)
        return l3
        "*** END YOUR CODE HERE ***"

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** BEGIN YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)
        "*** END YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** BEGIN YOUR CODE HERE ***"
        loss = 1e99
        while (loss > self.lossThreshold):
            for x, y in dataset.iterate_once(self.batchSize):
                loss = nn.as_scalar(self.get_loss(x, y))
                gradients = nn.gradients(
                    self.get_loss(x, y), [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(gradients[0], self.learningRate)
                self.b1.update(gradients[1], self.learningRate)
                self.w2.update(gradients[2], self.learningRate)
                self.b2.update(gradients[3], self.learningRate)
                self.w3.update(gradients[4], self.learningRate)
                self.b3.update(gradients[5], self.learningRate)
        "*** END YOUR CODE HERE ***"


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** BEGIN YOUR CODE HERE ***"
        self.batchSize = 20
        self.learningRate = -0.1
        self.w1 = nn.Parameter(784, 250)
        self.b1 = nn.Parameter(1, 250)
        self.w2 = nn.Parameter(250, 100)
        self.b2 = nn.Parameter(1, 100)
        self.w3 = nn.Parameter(100, 10)
        self.b3 = nn.Parameter(1, 10)
        # self.w4 = nn.Parameter(98, 10)
        # self.b4 = nn.Parameter(1, 10)
        "*** END YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** BEGIN YOUR CODE HERE ***"
        l1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        l2 = nn.ReLU(nn.AddBias(nn.Linear(l1, self.w2), self.b2))
        # l3 = nn.ReLU(nn.AddBias(nn.Linear(l2, self.w3), self.b3))
        l3 = nn.AddBias(nn.Linear(l2, self.w3), self.b3)
        return l3
        "*** END YOUR CODE HERE ***"

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** BEGIN YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)
        "*** END YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** BEGIN YOUR CODE HERE ***"
        accuracy = 0.0
        while (accuracy < 0.975):
            for x, y in dataset.iterate_once(self.batchSize):
                gradients = nn.gradients(
                    self.get_loss(x, y), [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(gradients[0], self.learningRate)
                self.b1.update(gradients[1], self.learningRate)
                self.w2.update(gradients[2], self.learningRate)
                self.b2.update(gradients[3], self.learningRate)
                self.w3.update(gradients[4], self.learningRate)
                self.b3.update(gradients[5], self.learningRate)
                # self.w4.update(gradients[6], self.learningRate)
                # self.b4.update(gradients[7], self.learningRate)
                accuracy = dataset.get_validation_accuracy()
        "*** END YOUR CODE HERE ***"


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** BEGIN YOUR CODE HERE ***"
        self.batchSize = 100
        self.learningRate = -0.1
        self.w1 = nn.Parameter(self.num_chars, 200)
        self.b1 = nn.Parameter(1, 200)
        self.w1I = nn.Parameter(self.num_chars, 200)
        self.b1I = nn.Parameter(1, 200)
        self.w2 = nn.Parameter(200, 200)
        self.b2 = nn.Parameter(1, 200)
        self.w3 = nn.Parameter(200, len(self.languages))
        self.b3 = nn.Parameter(1, len(self.languages))
        # self.w4 = nn.Parameter(98, 5)
        # self.b4 = nn.Parameter(1, 5)
        "*** END YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** BEGIN YOUR CODE HERE ***"
        node = nn.ReLU(nn.AddBias(
            nn.Linear(xs[0], self.w1I), self.b1I))
        for i in range(len(xs)):
            node = nn.ReLU(nn.AddBias(
                nn.Add(nn.Linear(xs[i], self.w1), nn.Linear(node, self.w2)), self.b2))
        return nn.AddBias(nn.Linear(node, self.w3), self.b3)
        "*** END YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** BEGIN YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)
        "*** END YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** BEGIN YOUR CODE HERE ***"
        loss = 1e99
        accuracy = 0.0
        while (accuracy < 0.895):
            for x, y in dataset.iterate_once(self.batchSize):
                gradients = nn.gradients(
                    self.get_loss(x, y), [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(gradients[0], self.learningRate)
                self.b1.update(gradients[1], self.learningRate)
                self.w2.update(gradients[2], self.learningRate)
                self.b2.update(gradients[3], self.learningRate)
                self.w3.update(gradients[4], self.learningRate)
                self.b3.update(gradients[5], self.learningRate)
                # self.w4.update(gradients[6], self.learningRate)
                # self.b4.update(gradients[7], self.learningRate)
                accuracy = dataset.get_validation_accuracy()
        "*** END YOUR CODE HERE ***"
