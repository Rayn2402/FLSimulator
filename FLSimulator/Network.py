"""

File containing both Node and Network classes

"""


class Node:

    __counter = 0

    def __init__(self, X, t):

        """
        Node of a federated network

        :param X: N x M numpy array of original features
        :param t: N x 1 numpy array of labels associated with the features
        """

        self.X, self.t = X, t
        self.n = self.X.shape[0]
        self.model = None
        self.id = 'Node ' + str(Node.__counter)
        Node.__counter += 1

    def __repr__(self):
        return self.id

    def update(self, E, C, w):

        """
        Train the node's model with local database

        :param E: number of epochs to execute
        :param C: batch size during training
        :param w: weight initial point in training

        """
        if self.model is None:
            raise Exception('Node has no model to train')

        self.model.train(self.X, self.t, E, C, weight_init=w)


class FederatedNetwork:

    def __init__(self, central_server, node_list):

        """
        Federated Network for federated learning

        :param central_server: object of class CentralServer
        :param node_list: list of nodes in the network

        """

        self.server = central_server
        self.nodes = node_list
        self.server.init_global_model_weights(self.nodes)
        self.server.copy_global_model(self.nodes)

    def run_learning(self, nb_of_rounds=1, show_round_results=False):

        """
        Run the federated learning

        :param nb_of_rounds: Rounds of federated learning to do
        :param show_round_results: tuple (bool, start, stop)
        """

        for i in range(nb_of_rounds):
            self.server.train(self.nodes)
            if show_round_results:
                self.global_accuracy(title='Round ' + str(i + 1))

    def global_accuracy(self, title=None):

        """
        Plots the model result over the complete network dataset
        Only available if the model as the function plot_model implemented (1-D or 2-D model)

        :param title: title of the figure
        """

        self.server.plot_global_accuracy(self.nodes, title)
