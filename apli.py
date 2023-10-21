import streamlit as st
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.model = Sequential()

    # Inside the NeuralNetwork class
    def add_layer(self, neurons, input_dim=None):
        if input_dim:
            self.model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
        else:
            self.model.add(Dense(neurons, activation='relu'))

    def compile_model(self):
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create the user interface with Streamlit
st.title('Neural Network Application MP')

# Allow the user to specify the number of layers and neurons
num_layers = st.slider('Number of layers', min_value=1, max_value=10)
neurons_per_layer = st.slider('Neurons per layer', min_value=1, max_value=100)

# Allow the user to specify the number of epochs
num_epochs = st.number_input('Number of epochs', min_value=0, max_value=1000, value=10)

# Create the neural network
nn = NeuralNetwork()
nn.add_layer(neurons=neurons_per_layer, input_dim=2)
for _ in range(num_layers - 1):  # Add the remaining layers
    nn.add_layer(neurons=neurons_per_layer)

if st.button('Compile'):
    nn.compile_model()  # Compile the model

    # Generate the make_moons dataset
    X, y = make_moons(n_samples=1000, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model with the number of epochs specified by the user
    history = nn.model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))

    # Create a grid of points to visualize classification regions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Predict the class of each point on the grid
    Z = nn.model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Show the model accuracy in a plot
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Training', 'Validation'], loc='upper left')

    # Show the classification regions
    axs[1].contourf(xx, yy, Z, alpha=0.8)
    scatter = axs[1].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    axs[1].set_title('Classification Regions')
    st.write('Model compiled successfully')
    # Show the plots
    st.pyplot(fig)


