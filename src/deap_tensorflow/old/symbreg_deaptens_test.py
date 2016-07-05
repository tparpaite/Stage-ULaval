from symbreg_deap_tensorflow import deap_run

# Creation des donnees artificielles pour test avec TensorFlow
data_train = []
for i in range(-100,100):
    x = i/10.0
    y = x**4 + x**3 + x**2 + 3.14 * x
    data_train.append([x, y])

data_test = []
for i in range(-10,10):
    x = i/10.0
    y = x**4 + x**3 + x**2 + 3.14 * x
    data_test.append([x, y])

data_train = numpy.array(data_train)
data_test = numpy.array(data_test)
