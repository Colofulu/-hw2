import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from art.attacks.evasion import FastGradientMethod
from art.defences.preprocessor import FeatureSqueezing
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainer
import tensorflow as tf

# 禁用 eager execution
tf.compat.v1.disable_eager_execution()

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define a simple CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create a KerasClassifier from the model
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Create an attack (Fast Gradient Method)
attack = FastGradientMethod(estimator=classifier, eps=0.1)

# Generate adversarial examples
x_train_adv = attack.generate(x_train, y=y_train)
x_test_adv = attack.generate(x_test, y=y_test)

# Define a defense (Feature Squeezing)
defense = FeatureSqueezing(clip_values=(0, 1), bit_depth=8)

# Preprocess the data using the defense
x_train_defense = defense(x_train_adv)
x_test_defense = defense(x_test_adv)

# Create an AdversarialTrainer for adversarial training
adv_trainer = AdversarialTrainer(classifier, attacks=attack)

# Train the classifier with adversarial examples
adv_trainer.fit(x_train, y_train, nb_epochs=5, batch_size=64)

# Evaluate the classifier on the test set
accuracy = classifier._model.evaluate(x_test, y_test)[1]
print(f"Accuracy on clean test data: {accuracy * 100:.2f}%")

accuracy_adv = classifier._model.evaluate(x_test_adv, y_test)[1]
print(f"Accuracy on adversarial test data: {accuracy_adv * 100:.2f}%")
