"""

File containing both Node and Network classes

"""


class Node:

    def __init__(self, DataGenerator):

        """
        Node of a federated network

        :param DataGenerator: object that generates data
        """

        self.DG = DataGenerator
        self.X, self.t = self.DG.generate_data()
        self.n = self.X.shape[0]
        self.model = None

    def train(self, E, C, w):

        """
        Train the node's model with local database

        :param E: number of epochs to execute
        :param C: batch size during training
        :param w: weight initial point in training

        """
        if self.model is None:
            raise Exception('Node has no model to train')

        self.model.train(self.X, self.t, E, C, w)
