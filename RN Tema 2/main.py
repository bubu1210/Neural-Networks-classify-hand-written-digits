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


def training(train_set, cifra):
    train_set_x, train_set_y = train_set
    # indexam train_y
    # t = numpy.eye(10)[train_set_y]
    learning_rate = 0.05
    nrIterations = 1
    allClassified = False
    b = numpy.random.uniform(0, 1, 1)
    w = numpy.random.uniform(0, 1, 784)
    target = numpy.eye(10)
    while not allClassified and nrIterations > 0:
        for i in range(0, len(train_set_x)):
            t = train_set_y[i]
            if t == cifra:
                t = 1
            else:
                t = 0
            # t = target[t][cifra]
            # z = w * x + b
            z = numpy.add(numpy.dot(w, train_set_x[i]), b)
            # if i == 0:
            #     print("z este: ", z)
            output = activation(z)
            # if i == 0:
            #     print("Output este: ", output)
            # w = w + (t[i] - output) * train_set_x[i] * learning_rate
            w = numpy.add(w, numpy.dot(numpy.dot((t - output), train_set_x[i]), learning_rate))
            # if i == 0:
            #     print("w-urile sunt: ", w)
            # b = b + (t[i] - output) * learning_rate
            b = numpy.add(b, (numpy.dot((t - output), learning_rate)))
            # if i == 0:
            #     print("b-ul este: ", b)
            if output != t:
                allClassified = False
        nrIterations -= 1
    return w, b


def vector_perceptroni(train_set):
    perceptroni = []
    for cifra in range(0, 10):
        perceptron_cifra = training(train_set, cifra)
        perceptroni.append(perceptron_cifra)
    return perceptroni


# print(training(train_set, 0))
# per = vector_perceptroni(train_set)
# print(per[1][1])
# print(training(train_set, 0))
# print(len(training(train_set, 1)[0]))
# print(len(training(train_set, 1)[1]))


def check_test(test_set):
    test_set_x, test_set_y = test_set
    set_perceptroni = vector_perceptroni(train_set)
    rights = numpy.zeros(10)
    total = numpy.zeros(10)
    for i in range(0, len(test_set_x)):
        x = test_set_x[i]
        t = test_set_y[i]
        perceptron = set_perceptroni[t]
        w = perceptron[0]
        b = perceptron[1]
        z = numpy.add(numpy.dot(w, x), b)
        output = activation(z)
        if output == 1:
            rights[t] += 1
        total[t] += 1

    for i in range(0, 10):
        print("Perceptronul ", i, "are randament= ", (rights[i]/total[i]) * 100 )


check_test(test_set)

