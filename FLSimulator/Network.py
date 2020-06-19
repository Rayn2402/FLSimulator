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
        self.n_k = self.X.shape[0]
        self.model = None
        self.p_k = None
        self.id = 'Node ' + str(Node.__counter)
        Node.__counter += 1

    def __repr__(self):
        return self.id

    def set_mass(self, n):

        """
        Set the mass of the node according to the ratio n_k / n

        :param n: total sample size of the entire network
        """
        self.p_k = self.n_k/n

    def update(self, E, C, w):

        """
        Trains the node's model with local database

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

        # We set the central server and the nodes
        self.server = central_server
        self.server.set_node_number(node_list)
        self.nodes = node_list

        # We init the global model and copy it in all nodes
        self.server.init_global_model_weights(self.nodes)
        self.server.copy_global_model(self.nodes)

        # We set nodes masses
        self.__set_nodes_masses(node_list)

    def run_learning(self, nb_of_rounds=1, show_round_results=False, loss_progress=False):

        """
        Runs the federated learning

        :param nb_of_rounds: Rounds of federated learning to do
        :param show_round_results: bool indicating if we show global accuracy plot between each round
        :param loss_progress: bool indicating if we should return an history of loss progression
        """

        # Initialization of list containing loss progression
        loss_progression = []

        for i in range(nb_of_rounds):

            self.server.train(self.nodes)

            if show_round_results:
                loss_progression.append(self.server.plot_global_accuracy(self.nodes, title='Round ' + str(i+1)))

            elif not show_round_results and loss_progress:
                loss_progression.append(self.server.global_loss(self.nodes))

        if loss_progress:
            return loss_progression

    @staticmethod
    def __set_nodes_masses(node_list):

        """
        Sets nodes masses according to ratio n_k / n

        :param node_list: list of nodes

        """
        n = sum([node.n_k for node in node_list])

        for node in node_list:
            node.set_mass(n)








