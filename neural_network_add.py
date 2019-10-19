import numpy as np
import math


class NeuralNetwork(object):
    def __init__(self, biggest):
        np.random.seed(1)
        self.biggest = biggest

        # 3 by 1 matrix for weights
        self.synaptic_weights = 2 * np.random.random((2, 1)) - 1

    def normalize(self, x, deriv=False):
        if deriv:
            return (1 - math.tanh(x/(2 * self.biggest)) ** 2)/(self.biggest)
        return math.tanh(x/(2 * self.biggest))

    def train(self, training_input, training_output):
        # Forward pass
        output = self.normalize(self.think(training_input))
        error = self.normalize(training_output) - output
        adjustments = np.dot(training_input.T, error * self.normalize(output, True))
        self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = np.dot(inputs, self.synaptic_weights)

        return output


if __name__ == '__main__':
    m_t = str(input('Manually Train(y/n)?: '))
    manual_train = (m_t is 'y')

    max_num = int(input('Max value: '))
    neural_net = NeuralNetwork(max_num)
    print('Random Synaptic Weights:\n', neural_net.synaptic_weights)

    if manual_train:
        while True:
            A = int(str(input('First Number: ')))
            B = int(str(input('Second Number: ')))

            print('Network: {} + {} = {}'.format(A, B, neural_net.think(np.array([A, B]))))

            neural_net.train(np.array([[A, B]]), np.array([[A + B]]).T)
    else:
        for i in range(100):
            A = np.random.randint(0, max_num)
            B = np.random.randint(0, max_num)

            print('epoch: {} Network prediction: {} + {} = {}'.format(i, A, B, neural_net.think(np.array([A, B]))))

            neural_net.train(np.array([[A, B]]), np.array([[A + B]]).T)
