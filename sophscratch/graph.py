from queue import Queue

import graphviz
import numpy as np

from .core import *





class Graph:
    """Represents a computational graph
    """

    def __init__(self):
        """Construct Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self

    def visualize(self):

        vis_g = graphviz.Digraph(comment='Tensor graph')
        vis_g.graph_attr['rankdir'] = 'LR'
        vis_g.graph_attr['nodesep'] = '.1'
        vis_g.graph_attr['ranksep'] = '.2'
        vis_g.graph_attr['ratio'] = 'compress'
        vis_g.node_attr["style"] = "rounded"
        vis_g.node_attr["shape"] = "box"

        tensors = self.operations + self.placeholders + self.variables

        for tensor in tensors:
            curr_id = hex(id(tensor))
            curr_name = tensor.__class__.__name__
            vis_g.node(curr_id, curr_name)

        for tensor in traverse_postorder(J):
            if not getattr(tensor, "input_nodes", None):
                continue
            curr_id = hex(id(tensor))
            for i_node in tensor.input_nodes:
                in_id = hex(id(i_node))
                vis_g.edge(in_id, curr_id)

        return vis_g


class Session:
    """Represents a particular execution of a computational graph.
    """

    def run(self, operation, feed_dict={}):
        """Computes the output of an operation
        """

        # Perform a post-order traversal of the graph to bring the nodes into the right order
        nodes_postorder = traverse_postorder(operation)

        # Iterate all nodes to determine their value
        for node in nodes_postorder:

            if type(node) == placeholder:
                # Set the node value to the placeholder value from feed_dict
                node.output = feed_dict[node]
            elif type(node) == Variable:
                # Set the node value to the variable's value attribute
                node.output = node.value
            else:  # Operation
                # Get the input values for this operation from node_values
                node.inputs = [
                    input_node.output for input_node in node.input_nodes]

                # Compute the output of this operation
                node.output = node.compute(*node.inputs)

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)

        # Return the requested node value
        return operation.output


class GradientDescentOptimizer(Operation):

    def __init__(self, loss, learning_rate=None):
        self.loss = loss
        self.lr = learning_rate
        super().__init__()

    def compute(self):
        # Compute gradients
        grad_table = compute_gradients(self.loss)

        # Iterate all variables
        for node in grad_table:
            if type(node) == Variable:
                # Retrieve gradient for this variable
                grad = grad_table[node]

                # Take a step along the direction of the negative gradient
                node.value -= self.lr * grad

        return grad_table


def compute_gradients(loss):

    # grad_table[node] will contain the gradient of the loss w.r.t. the node's output
    grad_table = {}

    # The gradient of the loss with respect to the loss is just 1
    grad_table[loss] = 1

    # Perform a breadth-first search, backwards from the loss
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        # If this node is not the loss
        if node != loss:
            #
            # Compute the gradient of the loss with respect to this node's output
            #
            grad_table[node] = 0

            # Iterate all consumers
            for consumer in node.consumers:

                # Retrieve the gradient of the loss w.r.t. consumer's output
                lossgrad_wrt_consumer_output = grad_table[consumer]

                # Get the gradient of the loss with respect to all of consumer's inputs
                lossgrads_wrt_consumer_inputs = consumer.gradient(
                    lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    # If there is a single input node to the consumer, lossgrads_wrt_consumer_inputs is a scalar
                    grad_table[node] += lossgrads_wrt_consumer_inputs

                else:
                    # Otherwise, lossgrads_wrt_consumer_inputs is an array of gradients for each input node

                    # Retrieve the index of node in consumer's inputs
                    node_index_in_consumer_inputs = consumer.input_nodes.index(
                        node)

                    # Get the gradient of the loss with respect to node
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]

                    # Add to total gradient
                    grad_table[node] += lossgrad_wrt_node

        #
        # Append each input node to the queue
        #
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    # Return gradients for each visited node
    return grad_table


def traverse_postorder(operation):
    """Performs a post-order traversal, returning a list of nodes
    in the order in which they have to be computed

    Args:
       operation: The operation to start traversal at
    """

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder
