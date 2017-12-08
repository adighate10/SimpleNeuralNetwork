from numpy import exp, array, random, dot


class PythonNN:
    def __init__(self):
        random.seed(1)

        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def __sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)

            error = training_set_outputs - output

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    neural_network = PythonNN()

    print("Random starting synaptic weights:")
    print(neural_network.synaptic_weights)

    training_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_inputs, training_outputs, 10000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    print("Considering new situation [1, 0, 0] -> ?:")
    op_one = neural_network.think(array([1, 0, 0]))
    print(op_one)

    print("Considering new situation [0, 0, 0] -> ?:")
    op_two = neural_network.think(array([0, 1, 0]))
    print(op_two)



