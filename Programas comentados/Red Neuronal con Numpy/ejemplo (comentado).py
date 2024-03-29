import mnist_loader
import network
import pickle

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=network.Network([784,30,10]) #Se establece el número de neuronas que hay en cada capa.
#La capa 1 corresponde a los 28x28 pixeles de cada imagen, la capa 2 se elige que sea de 30 neuronas y la capa 3 son los 10 números posibles.

net.SGD( training_data, 30, 10, 3.0, test_data=test_data)

archivo = open("red_prueba1.pkl",'wb') #pkl: Archivo-extensión de la librería Pickle: Vacía todo el contenido del objeto en ese archivo 
#(Ya no se queda guardado en la memoria, sino que ahora se guarda en el disco).
pickle.dump(net,archivo)
archivo.close()
exit()
#leer el archivo

archivo_lectura = open("red_prueba.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()

net.SGD( training_data, 10, 50, 0.5, test_data=test_data)

archivo = open("red_prueba.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()

#esquema de como usar la red :
imagen = leer_imagen("disco.jpg")
print(net.feedforward(imagen))
