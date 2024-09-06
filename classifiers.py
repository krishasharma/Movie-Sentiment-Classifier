import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """
    A classifier that always predicts the label '0' regardless of the input.
    """
    def predict(self, X):
        """
        Predict the labels for the given input data.

        Args:
            X (array): Feature matrix, ignored in this model as the prediction is constant.

        Returns:
            list: List filled with 0s, length equal to the number of samples in X.
        """
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(BinaryClassifier):
    """
    Naive Bayes Classifier implementing the probabilistic Naive Bayes algorithm for binary classification.
    """
    def __init__(self):
        """Initialize the Naive Bayes classifier with default values."""
        # Add your code here!
        super().__init__()
        self.log_prior = None
        self.log_likelihoods = None
        self.vocab_size = None
        

    def fit(self, X, Y):
        """
        Train the Naive Bayes classifier by estimating the log probabilities of the features based on the labels.

        Args:
            X (array): Feature matrix for training, shape (N, D).
            Y (array): Corresponding labels for the training set, shape (N,).
        """
        # Add your code here!
        # Count the number of documents in each class
        n_doc = len(Y)  # Number of documents in the training set
        n_positive = np.sum(Y)  # Number of positive (1) labels
        n_negative = n_doc - n_positive  # Number of negative (0) labels

        # Calculate log prior probabilities
        self.log_prior = {
            'positive': np.log(n_positive / n_doc),
            'negative': np.log(n_negative / n_doc)
        }

        # Calculate the total number of words in each class and the frequency of each word
        positive_counts = np.sum(X[Y == 1], axis=0)
        negative_counts = np.sum(X[Y == 0], axis=0)
        self.vocab_size = X.shape[1]

        # Apply add-1 smoothing to avoid zero probabilities
        self.log_likelihoods = {
            'positive': np.log((positive_counts + 1) / (np.sum(positive_counts) + self.vocab_size)),
            'negative': np.log((negative_counts + 1) / (np.sum(negative_counts) + self.vocab_size))
        }
        
    
    def predict(self, X):
        """
        Predict class labels for samples in X using the log probabilities computed during training.

        Args:
            X (array): Feature matrix for prediction, shape (N, D).

        Returns:
            array: Predicted class labels, shape (N,).
        """
        # Add your code here!
        # Calculate log probabilities for each class
        log_probs_positive = X @ self.log_likelihoods['positive'] + self.log_prior['positive']
        log_probs_negative = X @ self.log_likelihoods['negative'] + self.log_prior['negative']

        # Compare and predict the class with the higher log probability
        return np.where(log_probs_positive > log_probs_negative, 1, 0)


# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """
    Logistic Regression Classifier for binary classification using gradient descent optimization.
    """
    def __init__(self, learning_rate=0.01, n_iter=1000, lambda_reg=0.01):
        """Initialize the Logistic Regression classifier with specified learning rate, iterations, and regularization."""
        # Add your code here!
        super().__init__()
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.n_iter = n_iter  # Number of iterations for gradient descent
        self.lambda_reg = lambda_reg  # Regularization parameter
        self.weights = None  # Model weights
        

    def fit(self, X, Y):
        """
        Fit the Logistic Regression model using the training data.

        Args:
            X (array): Feature matrix for training, shape (N, D).
            Y (array): Labels for training, shape (N,).
        """
        # Add your code here!
        # Initialize weights
        self.weights = np.zeros(X.shape[1])  # Initialize weights to zeros
        for _ in range(self.n_iter):
            predictions = self.predict_proba(X)  # Predict probabilities
            # Compute gradient of the loss function
            gradient = np.dot(X.T, predictions - Y) / len(Y)
            # Update weights with L2 regularization
            self.weights -= self.learning_rate * (gradient + self.lambda_reg * self.weights)

    def predict_proba(self, X):
        """
        Predict probabilities for class 1 using the logistic function.

        Args:
            X (array): Feature matrix for prediction, shape (N, D).

        Returns:
            array: Predicted probabilities for class 1, shape (N,).
        """
        return 1 / (1 + np.exp(-np.dot(X, self.weights)))
    
    def predict(self, X):
        """
        Predict class labels based on the predicted probabilities.

        Args:
            X (array): Feature matrix for prediction, shape (N, D).

        Returns:
            array: Predicted class labels, shape (N,).
        """
        # Add your code here!
        return (self.predict_proba(X) >= 0.5).astype(int)


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()