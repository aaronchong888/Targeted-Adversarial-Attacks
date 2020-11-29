# import necessary packages
import sys
import os
import argparse
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
print('TensorFlow version: ', tf.__version__)

def get_target_class_index(label):
    label = label.lower().replace('_', ' ')
    jsonPath = os.path.join(os.path.dirname(__file__), 'imagenet_index.json')
    with open(jsonPath) as json_file:
        mapping_dict = json.load(json_file)
    return mapping_dict.get(label, None)    # default to None when Key not found

def preprocess_image(image):
	# swap color channels, resize the image, and add in a batch dimension
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image

def clip_eps(tensor, eps):
	# clip the values of input tensor to the range [-eps, eps]
	return tf.clip_by_value(tensor, clip_value_min = -eps, clip_value_max = eps)

def generate_target_adversaries(model, base_image, delta, original_class_index, target_class_index, steps):
	for step in range(0, steps):
		with tf.GradientTape() as tape:
			tape.watch(delta)
            # add our perturbation vector to the base image and preprocess the resulting image
			adversary = preprocess_input(baseImage + delta)
			# run this newly constructed image through the model and calculate the loss below
			predictions = model(adversary, training=False)
            # compute the negative sparse categorical cross-entropy loss with respect to the *original* class index
			originalLoss = -sccLoss(tf.convert_to_tensor([int(original_class_index)]), predictions)
            # compute the positive sparse categorical cross-entropy loss with respect to the *target* class index
			targetLoss = sccLoss(tf.convert_to_tensor([int(target_class_index)]), predictions)
			totalLoss = originalLoss + targetLoss
			# display the loss every 10 steps
			if step % 10 == 0:
				print('step: {}, loss: {}'.format(step, totalLoss.numpy()))
        
		# calculate the gradients of loss with respect to the perturbation vector
		gradients = tape.gradient(totalLoss, delta)

		# update the weights, clip the perturbation vector, and update its value
		optimizer.apply_gradients([(gradients, delta)])
		delta.assign_add(clip_eps(delta, eps=epsilon))

	return delta

# main
if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('file_in', help='input file name')
    parser.add_argument('target_class', help='target class name')
    args = parser.parse_args()

    target_index = get_target_class_index(args.target_class)
    if target_index == None:
        print('Error: Target class does not exist in ImageNet.')
        sys.exit(0)

    print('Input Filename: ', args.file_in)
    print('Target Class: ', args.target_class)
    print('Target Class Index: ', target_index)

    # preprocess the input image
    input_image = preprocess_image(cv2.imread(args.file_in))
    preprocessed_input = preprocess_input(input_image)

    # load the pre-trained ResNet50 model with ImageNet weights
    model = ResNet50(weights = 'imagenet')
    
    # make predictions on the input image and return the top 3 results
    predictions = model.predict(preprocessed_input)
    predictions = decode_predictions(predictions, top=3)[0]
    print('Top 3 Predictions: ', predictions)
    original_label = predictions[0][1]
    original_confidence = predictions[0][2]
    original_index = get_target_class_index(original_label)
    print('Original Class Index: ', original_index)

    # define the epsilon, learning rate and step number values
    epsilon = 0.01
    learning_rate = 0.01
    num_of_steps = 300

    # initialize optimizer and loss function
    optimizer = Adam(learning_rate = learning_rate)
    sccLoss = SparseCategoricalCrossentropy()
    
    # create a tensor based off the input image and initialize the perturbation vector (we will update this vector via training)
    baseImage = tf.constant(input_image, dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

    # generate the perturbation vector to create an adversarial example
    deltaUpdated = generate_target_adversaries(model, baseImage, delta, original_index, target_index, num_of_steps)

    # create the adversarial example, swap color channels, and save the output image to disk
    adverImage = (baseImage + deltaUpdated).numpy().squeeze()
    adverImage = np.clip(adverImage, 0, 255).astype('uint8')
    adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.png', adverImage)

    # make predictions again using the generated adversarial example
    preprocessed_input = preprocess_input(baseImage + deltaUpdated)
    predictions = model.predict(preprocessed_input)
    predictions = decode_predictions(predictions, top=3)[0]
    print('Top 3 Predictions on Adversarial Image: ', predictions)
    adv_label = predictions[0][1]
    adv_confidence = predictions[0][2]

    # display final results
    fig = plt.figure()
    panel1 = fig.add_subplot(1, 2, 1)
    panel1.set_title('{} \n\n {:.2f}% Confidence'.format(original_label, original_confidence * 100))
    panel1.imshow(mpimg.imread(args.file_in))
    panel1.axis('off')

    panel2 = fig.add_subplot(1, 2, 2)
    panel2.set_title('{} \n\n {:.2f}% Confidence'.format(adv_label, adv_confidence * 100))
    panel2.imshow(mpimg.imread('output.png'))
    panel2.axis('off')

    plt.show()
