import os

class CharNetConfig:
    """
    A plain configuration object used in the CharNet class. It acts as a helper, so that
    dictionaries don't have to be accessed, but instead, a simple class-based interface
    can be used.
    This reduces the number of magic strings, while improving maintainability and
    allowing for auto-completion in compatible IDEs.
    """

    def update(self, config: dict):
        """
        Update a configuration dictionary into the self variable. Used to avoid access
        of hidden __dict__ attribute. Note that this function does overwrite previously
        set parameters.
        :param config: Configuration dictionary used to set object parameters.
        :return: None
        """
        for key, value in config.items():
            setattr(self, key, value)

    def check(self):
        """
        A function to check if the entered configuration allows for a model creation or
        not. If it does, it tries cleaning up the configuration to simplify accessing
        its attributes. In case a model can not be created, an error is raised
        containing instructions to repair the issue.
        :return: None
        """
        if self.batch_size < 1:
            raise UserWarning("Make sure each batch contains at least one element.")

        if not self.test_string:
            raise UserWarning("A test string is required as it's fed to the model when"
                              "the epoch ends.")

        if self.load_model:
            return

        if not self.neuron_list:
            if self.neurons_per_layer * self.layer_count < 1:
                raise UserWarning(f"{self.neurons_per_layer} and {self.layer_count} "
                                  f"were provided. Make sure both are greater than "
                                  f"zero or provide a neuron list.")
            self.neuron_list = [self.neurons_per_layer] * self.layer_count
        self.layer_count = len(self.neuron_list)
        if not self.block_depth:
            self.block_depth = [1] * self.layer_count
        elif isinstance(self.block_depth, int):
            self.block_depth = [self.block_depth] * self.layer_count
        elif len(self.block_depth) == 1:
            self.block_depth = [self.block_depth[0]] * self.layer_count
        else:
            raise UserWarning(f"Unable to handle block depth {self.block_depth} for "
                              f"{self.layer_count} layers. Make sure to either use only"
                              f"one element or to provide a list of the same length as "
                              f"layers.")
        if not self.classes and self.embedding:
            raise UserWarning(f"When using embedding, the number of classes predicted"
                              f"has to be greater than zero. You entered {self.classes}"
                              )
        if not self.output_activation:
            self.output_activation = None
        if not self.loss:
            self.loss = "mean_squared_error"

    def __init__(self, config=None, **kwargs):
        self.neurons_per_layer = 16
        self.layer_count = 4
        self.inputs = 16
        self.classes = 30
        self.dropout = 0.3
        self.input_dropout = 0.1
        self.batch_size = 1024
        self.learning_rate = 1e-4
        self.generated_characters = 512
        self.neuron_list = []
        self.block_depth = []
        self.metrics = ['accuracy']
        self.embedding = True
        self.class_neurons = True
        self.load_model = False
        self.output_activation = "softmax"
        self.loss = "sparse_categorical_crossentropy"
        self.model_folder = "mlp_weights-"
        with open(os.path.join(os.path.dirname(__file__), '..', 'LICENSE'), 'r') as f:
            self.test_string = f.read()

        if isinstance(config, dict):
            self.update(config)
        self.update(kwargs)

        self.check()
