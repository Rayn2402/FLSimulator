"""

File containing both Node and Network classes

"""


class Node:

    __counter = 0

    def __init__(self, DataGenerator):

        """
        Node of a federated network

        :param DataGenerator: object that generates data
        """

        self.DG = DataGenerator
        self.X, self.t = self.DG.generate_data()
        self.n = self.X.shape[0]
        self.model = None
        self.id = 'Node ' + str(Node.__counter)
        Node.__counter += 1

    def update(self, E, C, w):

        """
        Train the node's model with local database

        :param E: number of epochs to execute
        :param C: batch size during training
        :param w: weight initial point in training

        """
        if self.model is None:
            raise Exception('Node has no model to train')

        self.model.train(self.X, self.t, E, C, w)


class FederatedNetwork:

    def __init__(self, central_server, node_list):

        """
        Federated Network for federated learning

        :param central_server: object of class CentralServer
        :param node_list: list of nodes in the network

        """

        self.server = central_server
        self.nodes = node_list
        self.server.copy_global_model(self.nodes)

    def run_learning(self, nb_of_rounds=1):

        """
        Run the federated learning

        :param nb_of_rounds: Rounds of federated learning to do
        """

        for i in range(nb_of_rounds):
            self.server.train(self.nodes)

    def global_accuracy(self, start, stop):

        """
        Plots the model result over the complete network dataset
        Only available if the model as the function plot_model implemented (1-D or 2-D model)

        :param start: start on x-axis
        :param stop: stop on x-axis
        """

        self.server.plot_global_accuracy(self, self.nodes, start, stop)
