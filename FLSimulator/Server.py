"""

File containing CentralServer class that dispatches training of global model in the original
federated learning model

"""
import numpy as np
import random
import copy

aggregation_choices = ['FedAvg']
node_selection_choices = ['all', 'random']


class CentralServer:

    def __init__(self, global_model, aggregation, node_selection, C=1, E=100, random_size=0.80):

        """
        Central server that that dispatches training of global model in the federated learning network

        :param global_model: global learning model shared by the server to the federated network
        :param aggregation: aggregation function choice
        :param node_selection: node selection choice
        :param C: local mini-batch size in every node training
        :param E: local number of epochs made in every node training
        :param random_size: percentage of node selected when node selection choice is not 'all'
        """
        self.global_model = global_model
        self.C = C
        self.E = E
        self.select_nodes = self.node_selection_function(node_selection, random_size)
        self.aggregate = self.aggregation_function(aggregation)

    @staticmethod
    def node_selection_function(node_selection, random_size):

        """
        Build the function to select node in list of nodes available

        :param node_selection: node selection choice
        :param random_size: percentage of randomly selected nodes if node selection is 'random'
        :return: function
        """
        if node_selection not in node_selection_choices:
            raise Exception('Node selection choice must be in {}'.format(node_selection_choices))

        if node_selection == 'all':

            def select(node_list): return node_list

        else:

            if random_size >= 1 or random_size <= 0:
                raise BaseException('random size must be between 0 and 1 excluded')

            def select(node_list): return random.sample(node_list, round(random_size*len(node_list)))

        return select

    def aggregation_function(self, aggregation):

        """
        Build the function to aggregate models from a list of nodes

        :param aggregation: aggregation choice
        :return: function
        """

        if aggregation not in aggregation_choices:
            raise Exception('Aggregation choice must be in {}'.format(aggregation_choices))

        def aggregate(node_list):

            # We compute the total sample size every time in case nodes are randomly sampled in each round
            # of federated training
            N = sum([node.n for node in node_list])

            # We update w with a weighted average of every node models w
            self.global_model.w = sum([(node.n/N)*node.model.w for node in node_list])

        return aggregate

    def copy_global_model(self, node_list):

        """
        Copy global model in each node of the node list
        Used in the initialization of a Network

        :param node_list: list of nodes
        """
        for node in node_list:
            node.model = self.global_model.copy()

    def train(self, node_list):

        """
        Train the global model among a list of nodes

        :param node_list: list of nodes
        """
        selected_node_list = self.select_nodes(node_list)

        for node in selected_node_list:
            node.update(self.E, self.C, copy.deepcopy(self.global_model.w))

        self.aggregate(selected_node_list)

    def init_global_model_weights(self, node_list):

        """
        Initializes the weights of the global model

        :param node_list: list of nodes
        """
        input_sample = node_list[0].X[0:1]
        feature_size = self.global_model.phi(input_sample).shape[1]
        self.global_model.init_weight(feature_size)

    def plot_global_accuracy(self, node_list, title=None):

        """
        Plots the model result over the complete network dataset
        Only available if the model as the function plot_model implemented (1-D or 2-D model)

        :param node_list: list of nodes
        :param title: title of the figure
        """

        loss, X_total, t_total = self.global_loss(node_list, grouped_data=True)

        if title is None:
            title = 'Loss : ' + str(loss)
        else:
            title += ' - Loss : ' + str(loss)

        self.global_model.plot_model(X_total, t_total, title)

        return loss

    def global_loss(self, node_list, grouped_data=False):

        """
        Computes the sum of local loss of different models and loss of global model among all nodes dataset

        :param node_list: list of nodes
        :return: weighted loss and total loss
        """
        # We compute the total sample size
        N = sum([node.n for node in node_list])

        # We compute sum of global loss
        X_total, t_total = regroup_data_base(node_list)
        loss = round(self.global_model.loss(X_total, t_total), 2)

        if grouped_data:
            return loss, X_total, t_total
        else:
            return loss


def regroup_data_base(node_list):

    """
    Regroups all data base of the different Nodes into a single data base

    :param node_list: list of Nodes
    :return: X, t : numpy array
    """

    X_total = node_list[0].X
    t_total = node_list[0].t

    if len(node_list) >= 1:
        for node in node_list[1:]:
            X_total = np.append(X_total, node.X, axis=0)
            t_total = np.append(t_total, node.t, axis=0)

    return X_total, t_total

