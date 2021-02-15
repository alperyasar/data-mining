import numpy as np
import scipy.special as sc



class LogisticRegression(object):

    def __init__(self, gd_iterations, number_of_features_to_select, online_learning_rate):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.w = None
        self.online_learning_rate = online_learning_rate
        self.gd_iterations = gd_iterations
        self.number_of_features_to_select = number_of_features_to_select
        self.indices = None
        self.num_examples = None


    def fit(self, X, y):
        # TODO: Write code to fit the model.

        self.num_examples, self.num_input_features = X.shape
        if self.number_of_features_to_select != -1:
            X = self.feature_Selection(X, y)

        self.num_examples, self.num_input_features = X.shape

        self.w = np.zeros(shape=(1, self.num_input_features), dtype=float)
        prob = np.zeros(shape=(1, self.num_examples), dtype=float)
        prob_neg = np.zeros(shape=(1, self.num_examples), dtype=float)

        for i in range(self.gd_iterations):
            delta = np.zeros(shape=(1, self.num_input_features), dtype=float)
            prob = sc.expit(self.w@X.T)
            prob_neg = sc.expit(-1 * (self.w@X.T))

            delta = (np.multiply(y, prob_neg)) * X + \
                (np.multiply(1 - y, prob)) * (-1 * X)
            self.w += self.online_learning_rate * delta

    def predict(self, X):
        # TODO: Write code to make predictions.
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')

        try:
            if self.num_input_features != -1:
                X = X[:, self.indices]
        except:
            pass

        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        try:
            if self.num_input_features != -1:
                X = X[:, self.indices]
        except:
            pass

        y_hat = np.empty([num_examples], dtype=np.int)

        for i, row in enumerate(X.toarray()):
            row = row[np.newaxis]
            val = self.w@row.T
            prob = sc.expit(val)
            if prob >= 0.5:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        return y_hat
