class IndividualTensor:
    """ On l'utilise surtout pour memoriser les informations liees a un tenseur
    """

    __slots__ = ('individual', 'input', 'output', 'weights', 'tensor')

    def __init__(self, individual, input, output, weights, dict_arg):
        self.individual = individual
        self.input = input
        self.output = output
        self.weights = weights
        self.dict_arg = dict_arg
        self.tensor = None

    def __eq__(self):
        """Inutile pour le moment, TODO"""
        return NotImplemented

    def __str__(self):
        """Return the expression in a human readable string."""

        string = []

        string.append("Individual :", str(self.individual))
        string.append("\nInput :", self.input)
        string.append("\nOutput :", self.output)
        string.append("\nWeights :", self.weights)
        string.append("\nTensor :", self.tensor)

        return string

        
