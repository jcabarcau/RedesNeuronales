"""
network.py
~~~~~~~~~~
Módulo para implementar el algoritmo de aprendizaje 'Stochastic Gradient Descent' (SGD) para una red neuronal 'feedforward'.
Los gradientes se calculan usando el algoritmo 'backpropagation'.
(Según el autor, este código no está del todo optimizado y omite muchas características deseables).
"""

#### Librerías
# Librerías estándar
import random

# Librerías de terceros
import numpy as np #Librería que ayuda a interpretar los datos como una matríz V[x,y] (con x renglones y y filas), ayudando a manipular la información de los datos.
# Esta librería tiene que ser instalada individualmente en el ordenador.
# En el caso de Windows, esta puede instalarse en el Bash de Git, o en Windows PowerShell, con el comando 'pip3 install numpy' (Para Python 3).

class Network(object): #Instrucciones para construir una Red Neuronal. 
    #Definimos la clase 'Network' como un objeto. Las variables en python son 'etiquetas' de los objetos.

    def __init__(self, sizes): #Definimos el método '__init__', el método de inicialización de una clase, ejecutandose automáticamente cuando se crea una nueva instancia de clase.
        #Recibe como argumento 'sizes', la lista que especifica el número de neuronas en cada capa de la red neuronal.
        """
        La lista 'sizes' (tamaños) contiene el número de neuronas en las respectivas capas de la red.
        Por ejemplo, si la lista fuera [2, 3, 1], entonces sería una red de 3 capas, donde la primera capa contendría 2 neuronas, la segunda 3, y la tercera 1.
        Los 'biases' (sesgos) y los 'weights' (pesos) para la red se inicializan aleatoriamente, utilizando una distribución gaussiana con media 0 y varianza 1.
        Tomar en cuenta que se supone que la primera entrada es una capa de 'inputs' (entradas) y, por convención, no se establecerá ningún 'bias' para esas neuronas,
            ya que los biases solo se utilizan para calcular los 'outputs' (salidas) de capas posteriores.
        """
        self.num_layers = len(sizes) #Almacena el número total de capas en la red neuronal
        self.sizes = sizes #Almacena la lista de tamaños de capa que se pasaron como argumento
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Crea una lista de matríces aleatorias que se utilizaran como biases para cada capa de red, excepto la primera.
        #En 'np.random.randn(y, 1)', el (y,1) es importante, ya que hace que Python interprete a la matriz como una matriz de 1 columna.
        #En 'sizes[1:]', el 1: significa 'empezar a leer con el primer índice, de izquierda a derecha' (o empieza en el segundo elemento?) [confirmar].
        self.weights = [np.random.randn(y, x) #Crea una lista de matrices aleatorias que se utilizan como pesos para cada conexión de las capas de la red.
                        #Cada matriz tiene dimensiones (número de neuronas en la capa siguiente, número de neuronas en la capa actual).
                        for x, y in zip(sizes[:-1], sizes[1:])] #'zip' (cierre): Función que sirve para 'pegar' las listas en una sola matriz.
                        #En 'sizes[:-1]', el :-1 significa 'empezar a leer de izquierda a derecha, terminando en el penúltimo elemento' (?) [confirmar también].

    def feedforward(self, a): #Se define la función 'feedforward', cuyo primer parámetro es 'self', que hace referencia a la instancia de clase,
        #el segundo parámetro es 'a', que representa la entrada de la red neuronal.
        """Devuelve el output de la red si se ingresa un input 'a'."""
        for b, w in zip(self.biases, self.weights): #Bucle que itera sobre las listas 'self.biases' y 'self.weights' simultáneamente usando la función 'zip', la cual las une.
            #'b' y 'w' representan los biases y los pesos asociados a una capa específica de la red neuronal.
            a = sigmoid(np.dot(w, a)+b) #Bucle que realiza la operación de propagación hacia adelante para cada capa de la red neuronal.
            #'np.dot(w,a)' calcula el producto punto entre los pesos 'w' y el input 'a', sumando el bias 'b'.
            #La función 'sigmoid' (sigmoide) aplica la función de activación sigmoide a esta suma ponderada. 
        return a #Devuelve el output de la red.

    def SGD(self, training_data, epochs, mini_batch_size, eta, #Se define el algoritmo SGD (Stochastic Gradient Descent).
            #Se usa para entrenar la red neuronal. Recibe como primer argumento los datos de entrenamiento,
            # después el número de épocas, después el tamaño de los mini batches, y al final la tasa de aprendizaje (learning rate), 'eta'.
            test_data=None):
        """Entrena la red neuronal usando el Stochastic Gradient descent de mini-batches.
        El 'training_data' es una lista de tuplas '(x,y)' que representan los training inputs y los outputs deseados.
        Los demás parámetros no opcionales se explican por sí solos.
        Si se proporciona 'test_data', la red se evaluará con los test data después de cada época y se imprimirá el progreso parcial.
        Esto es útil para seguir el progreso, pero ralentiza considerablemente el proceso."""
        if test_data: #Verifica si se proporcionaron los datos de prueba.
            test_data = list(test_data) #Los datos de prueba proporcionados se convierten en una lista.
            n_test = len(test_data) #Almacena la longitud del conjunto de datos en 'n_test'.

        training_data = list(training_data) #Crea una lista con los datos de entrenamiento.
        n = len(training_data) #Almacena la longitud del conjunto de datos de entrenamiento en la variable 'n'.
        for j in range(epochs): #Bucle que se ejecuta un número específico de épocas.
            random.shuffle(training_data) #Reorganiza los datos de entrenamiento aleatoriamente. Esto garantiza que la red neuronal no se entrene en el mismo orden en cada época.
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] #Divide el conjunto de datos de entrenamiento en mini-batches del tamaño especificado 'mini_batch_size'.
            for mini_batch in mini_batches: #Itera sobre cada minibatch.
                self.update_mini_batch(mini_batch, eta) #Llama al método 'update_mini_batch', el cual actualiza los pesos y bias de la red neuronal utilizando el mini-batch
                #actual y el learning rate 'eta'.
            if test_data: #Verifica si se proporcionaron datos de prueba 
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)) #Imprime la cantidad de épocas en las que va la simulación. {0}, {1} y {2} actúan como marcadores de posición para los
                # valores que se insertarán después usando '.format(j, self.evaluate(test_data), n_test)', el cual sustituye {0} por 'j' (el número de la época actual),
                # {1} por 'self.evaluate(test_data)' (valor que representa la presición de la red neuronal en el conjunto de datos de prueba),
                # y {2} por 'n_test' (la longitud de los datos de prueba.)
            else:
                print("Epoch {0} complete".format(j)) #Imprime el número de época e indica que se ha completado la época actual del entrenamiento de la red neuronal.
                #A mayor learning rate, más alto es el número obtenido en la primera época,
                # pero si el learning rate es muy pequeño, puede ajustarse mejor a los datos de prueba, o puede tener un sobreajuste de datos.

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
