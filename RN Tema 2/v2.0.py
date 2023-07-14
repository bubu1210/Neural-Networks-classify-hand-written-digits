import gzip
import numpy
import pickle

from matplotlib import pyplot

with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')


# afisare img
def afisare_img(nr_imagine):
    pyplot.imshow(train_set[0][nr_imagine].reshape(28, 28))
    # print(train_set[0][nr_imagine].reshape(28, 28))
    print("Imaginea afisata de pe pozitia ", nr_imagine, " este: ", train_set[1][nr_imagine])
    pyplot.show()


# afisare_img(0)


def activation(inputt):
    if inputt > 0:
        return 1
    else:
        return 0


def training(train_set):
    train_set_x, train_set_y = train_set
    # indexam train_y
    t = numpy.eye(10)[train_set_y]
    learning_rate = 0.1
    nrIterations = 100
    allClassified = False
    b = numpy.zeros(10)
    w = numpy.random.uniform(0, 1, 784)
    w.reshape(28, 28)
    perceptroni = []


    while not allClassified and nrIterations > 0:
        for i in range(0, len(train_set)):
            for cifra in range(0, 10):
                perceptron_cifra = []
                # z = w * x + b
                z = numpy.add(numpy.dot(w.reshape(28, 28), train_set_x[i].reshape(28, 28)), b.reshape(10, 1))
                if i == 0:
                    print("z este", z)
                output = activation(z[cifra])
                if i == 0:
                    print("Output este ", output)
                # w = w + (t[i] - output) * train_set_x[i] * learning_rate
                w = numpy.add(w, numpy.dot(numpy.dot((t[i] - output), train_set_x[i]), learning_rate))
                if i == 0:
                    print("W este: ", w)
                # b = b + (t[i] - output) * learning_rate
                b = numpy.add(b, (numpy.dot((t[i] - output), learning_rate)))
                if i == 0:
                    print("B-urile sunt: ", b)
                if output != t[i]:
                    allClassified = False
        nrIterations -= 1
    return w, b


def vector_perceptroni(train_set):
    perceptroni = []
    for cifra in range(0, 10):
        perceptron_cifra = training(train_set)
        perceptroni.append(perceptron_cifra)
    return perceptroni


training(train_set)

# print(training(train_set, 0))
# print(len(training(train_set, 1)[0]))
# print(len(training(train_set, 1)[1]))

# def start(train_set, test_set):
