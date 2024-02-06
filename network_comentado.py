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
        #En resumen, este código implimenta el entrenamiento de la red neuronal utilizando el algoritmo SGD. Cada época implica 'barajar' los datos de entrenamiento,
        # dividirlos en mini-batches, y actualizar los pesos y biases de la red neuronal para MINIMIZAR LA FUNCIÓN DE COSTO.

    def update_mini_batch(self, mini_batch, eta): #Define la función 'update_mini_batch'.
        """Actualiza los pesos y biases de la red aplicando un gradient descent usando backpropagation para un único mini batch.
        El 'mini_batch' es una lista de tuplas '(x,y)' y 'eta' es la taza de aprendizaje."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Se inicializa 'nabla_b' como una lista de matrices de ceros, donde cada matriz tiene la misma forma que el bias
        # correspondiente en la red neuronal. Esto se utilizará para acumular los gradientes de la función de costo con respecto a los biases durante el backpropagation.
        nabla_w = [np.zeros(w.shape) for w in self.weights] #Similar a la línea anterior, pero para los pesos.
        for x, y in mini_batch: #Bucle que itera sobre cada ejemplo '(x,y)' en el mini-batch.
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #Llama al método 'backprop' para calcular los gradientes de la función de costo con respecto a los biases (delta_nabla_b)
            # y los pesos (delta_nabla_w) utilizando el ejemplo actual y su etiqueta correspondiente.
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #Acumula los gradientes de los biases para cada ejemplo en el mini-batch.
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #Acumula los gradientes de los pesos para cada ejemplo en el mini-batch.
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] #Actualiza los pesos de la red neuronal utilizando el gradient descent.
        #Para cada peso, se resta un término proporcional al gradiente del peso multiplicado por el learning rate y dividido por el tamaño del mini-batch.
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)] #Similar al paso anterior, pero actualiza los biases.
        #En resumen, este método se encarga de actualizar los pesos y los biases de la red neuronal, utilizando el método SGD en un mini-batch durante el entrenamiento.

    def backprop(self, x, y):
        """Devuelve una tupla '(nabla_b, nabla_w)', que representa el gradiente de la función de costo C_x.
        'nabla_b' y 'nabla_w' son listas capa por capa de matríces numpy, similares a 'self.biases' y 'self.weights'."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Se inicializa una lista 'nabla_b' con gradientes de la función de costo con respecto a los biases.
        # Cada elemento de la lista es una matriz de ceros con la misma forma que el bias correspondiente.
        nabla_w = [np.zeros(w.shape) for w in self.weights] #Similar a la línea anterior pero para los pesos.
        '''Feedforward'''
        activation = x #Se inicializa 'activation' con la entrada 'x'. Esta variable se utilizará para almacenar las activaciones de cada capa durante el proceso de backpropagation.
        activations = [x] '''Lista para almacenar todas las activaciones, capa por capa.'''
        zs = [] '''Lista para almacenar todos los vectores z, capa por capa.'''
        #[] significa que se le asigna una lista vacía a zs.
        for b, w in zip(self.biases, self.weights): #Se inicia un bucle que itera sobre cada bias 'b' y peso 'w' en la red neuronal.
            z = np.dot(w, activation)+b #Calcula la entrada ponderada de la neurona sumando el producto punto de los pesos 'w' y la activación anterior con el bias 'b'.
            zs.append(z) #Agrega la entrada ponderada 'z' a la lista 'zs'.
            activation = sigmoid(z) #Calcula la activación de la neurona aplicando la función de activación sigmoide a la entrada ponderada 'z'.
            activations.append(activation) #Agrega la activación 'activation'a la lista 'activations'.
        # Backward Pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) #Calcula el error en la capa de salida multiplicando la derivada de la función de costo con respecto a la activación 'self.cost_derivative'
            # por la derivada de la función de activación en la capa de salida 'sigmoid_prime'.
        nabla_b[-1] = delta #Asigna el error calculado a la última capa de biases en 'nabla_b'.
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #Calcula el gradiente de la función de costo con respecto a los pesos en la última capa y lo asigna a 'nabla_w[-1]'.
        '''Tenga en cuenta que la variable 'l' en el bucle siguiente se usa de manera un poco diferente a la notación del Capítulo 2 del libro.
        Aquí, l=1 significa la última capa de neuronas, l=2 es la penúltima capa, y así sucesivamente.
        Es una renumeración del esquema del libro, que se usa aquí para aprovechar el hecho de que Python puede usar índices negativos en las listas.'''
        for l in range(2, self.num_layers): #Inicia otro bucle que itera sobre las capas desde  la última hasta la segunda capa.
            z = zs[-l] #Obtiene la entrada ponderada de la capa actual.
            sp = sigmoid_prime(z) #Calcula la derivada de la función de activación en la capa actual.
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #Calcula el error en la capa actual propagando hacia atrás el error desde la capa siguiente,
            # multiplicando por la derivada de la función de activación en la capa actual.
            nabla_b[-l] = delta #Asigna el error calculado a la capa actual de biases en 'nabla_b'.
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #Calcula el gradiente de la función de costo con respecto a los pesos en la capa actual
            # y lo asigna a 'nabla_w[-1]'
        return (nabla_b, nabla_w) #Retorna las listas 'nabla_b' y 'nabla_w', que contienen los gradientes de la función de costo con respecto a los biases y pesos,
        #respectivamente, para cada capa de la red neuronal.
    #En resumen, este método implementa el algoritmo Backpropagation para calcular los gradientes de la función de costo con respecto a los biases y pesos de la red neuronal.
    #Estos gradientes se usan luego para actualizar los parámetros durante el entrenamiento mediante el método SGD.

    def evaluate(self, test_data): #Definimos la función 'evaluate' donde 'test_data' es una lista de tuplas de los datos de prueba, donde cada tupla contiene una entrada 'x'
        # y una salida deseada 'y'.
        """Devuelve el número de entradas de prueba para las cuales la red neuronal genera el resultado correcto.
        Tenga en cuenta que se supone que el output de la red neuronal es el índice de la neurona de la capa final que tenga mayor activación."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data] #Para cada entrada 'x' en el conjunto de datos de prueba, la red neuronal realiza una predicción utilizando la función 'feedforward'
        # y se compara la predicción con la salida deseada 'y'. Los resultados se almacenan en la lista 'test_results', donde cada elemento es una tupla de 'x' y 'y'.
        return sum(int(x == y) for (x, y) in test_results) #Retorna la suma de las coincidencias entre las predicciones de la red y las salidas deseadas en los datos de prueba.
        #La expresión 'int(x == y)' es 1 si 'x' es igual a 'y', y 0 en caso contrario. Por lo tanto, la función devuelve la cantidad total de predicciones correctas.

    def cost_derivative(self, output_activations, y): #Definimos la derivada parcial de la función de costo donde 'output_activations' representa las activaciones
        # de salida de la red neuronal, es decir, las predicciones de la red para un ejemplo de entrada. Y 'y' representa la salida deseada para el mismo ejemplo de entrada.
        """Devuelve el vector de derivadas parciales {\partial C_x}/{/partial a} para las activaciones de salida (output activations)."""
        return (output_activations-y) #Retorna la derivada parcial de la función de costo con respecto a las activaciones de salida.
        #Esta función se utiliza en el proceso de backpropagation para calcular los gradientes necesarios para ajustar los parámetros de la red neuronal.
    

#### Otras funciones
def sigmoid(z):
    """La función sigmoide."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivada de la función sigmoide."""
    return sigmoid(z)*(1-sigmoid(z))
