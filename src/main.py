import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from neural_network import NeuralNetwork


def main(argv):
	if len(argv) < 3:
	    sys.exit('Usage: %s training.csv test.csv' % argv[0])
	if not os.path.exists(argv[1]) or not os.path.exists(argv[2]):
	    sys.exit('ERROR: Game %s was not found!' % argv[1])

	training_file = open(argv[1], 'r')
	training_data = training_file.readlines()
	training_file.close()
	test_file = open(argv[2], 'r')
	test_data = test_file.readlines()
	test_file.close()

	n_input = 784
	n_hidden = 500
	n_output = 10
	learning_rate = 0.1
	epochs = 7

	nn = NeuralNetwork(n_input, n_hidden, n_output, learning_rate)
	
	for e in range(epochs):
		for record in training_data:
			img_values = record.split(',')
			inputs = (np.asfarray(img_values[1:]) / 255.0 * 0.99) + 0.01
			inputs_plus_10_degrees = ndi.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, reshape=False)
			inputs_minus_10_degrees = ndi.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, reshape=False)
			target = np.zeros(n_output) + 0.01
			target[int(img_values[0])] = 0.99
			nn.train(inputs, target)
			nn.train(inputs_plus_10_degrees.reshape(784), target)
			nn.train(inputs_minus_10_degrees.reshape(784), target)

	score = []
	for record in test_data:
		img_values = record.split(',')
		inputs = (np.asfarray(img_values[1:]) / 255.0 * 0.99) + 0.01
		correct_label = int(img_values[0])
		output = nn.query(inputs)
		nn_label = np.argmax(output)
		if nn_label == correct_label:
			score.append(1)

	print(np.asarray(score).sum() / len(test_data), 'performance')


if __name__ == "__main__":
    main(sys.argv)
