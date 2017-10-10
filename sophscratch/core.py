
class BaseTensor:

    def get_id(self):
        return hex(id(self))

    def get_name(self):
        return self.__class__.__name__


class placeholder(BaseTensor):

    def __init__(self):

        self.consumers = []
        self.tensor = "placeholder"


class Variable(BaseTensor):

    def __init__(self, initial_value=None):

        self.value = initial_value
        self.consumers = []
        self.tesnor = "variable"


class Operation(BaseTensor):
    """Represents a graph node that performs a computation
    """

    def __init__(self, input_nodes=[]):

        self.input_nodes = input_nodes
        self.consumers = []

        for input_node in input_nodes:
            input_node.consumers.append(self)

    def compute(self):
        """ Computes output of the operation
        Must be implemented by individual operations
        """
        pass

    def gradient(self, grad):
        """ Computes the gradient for the operation tensor
        """
        pass
