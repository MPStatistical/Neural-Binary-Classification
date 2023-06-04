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

    # Dentro de la clase NeuralNetwork
    def add_layer(self, neurons, input_dim=None):
        if input_dim:  
            self.model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
        else:
            self.model.add(Dense(neurons, activation='relu'))


    def compile_model(self):
        self.model.add(Dense(1, activation='sigmoid'))  # Capa de salida para clasificación binaria
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Crear la interfaz de usuario con Streamlit
# Crear la interfaz de usuario con Streamlit
st.title('Aplicación de Red Neuronal MP Statistical')

# Permitir al usuario especificar el número de capas y neuronas
num_layers = st.slider('Número de capas', min_value=1, max_value=10)
neurons_per_layer = st.slider('Neuronas por capa', min_value=1, max_value=100)

# Permitir al usuario especificar el número de épocas
num_epochs = st.number_input('Número de épocas', min_value=0, max_value=1000, value=10)

# Crear la red neuronal
nn = NeuralNetwork()
nn.add_layer(neurons=neurons_per_layer, input_dim=2) 
for _ in range(num_layers - 1):  # Añade las capas restantes
    nn.add_layer(neurons=neurons_per_layer)

if st.button('Compilar'):
    nn.compile_model()  # Compila el modelo
    st.write('Modelo compilado con éxito')

    # Generar el conjunto de datos make_moons
    X, y = make_moons(n_samples=1000, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo con el número de épocas especificado por el usuario
    history = nn.model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))

    # Crear una cuadrícula de puntos para visualizar las regiones de clasificación
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predecir la clase de cada punto en la cuadrícula
    Z = nn.model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Crear subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar la precisión del modelo en una gráfica
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Precisión del modelo')
    axs[0].set_ylabel('Precisión')
    axs[0].set_xlabel('Época')
    axs[0].legend(['Entrenamiento', 'Validación'], loc='upper left')

    # Mostrar las regiones de clasificación
    axs[1].contourf(xx, yy, Z, alpha=0.8)
    scatter = axs[1].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    axs[1].set_title('Regiones de clasificación')

    # Mostrar las gráficas
    st.pyplot(fig)

