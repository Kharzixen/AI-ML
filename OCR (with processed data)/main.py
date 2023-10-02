import math
from collections import Counter
import numpy as np

vectors = []


def ReadData():
    file = open('optdigits.tra', 'r')
    lines = file.readlines()
    for line in lines:
        v = np.fromstring(line, dtype=int, sep=',')
        vectors.append((v[64], v[:64]))


def EuclideanDistance(vector1, vector2):
    s = 0
    for i in range(0, 64):
        s += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i])
    return math.sqrt(s)


def CosinusSimilarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


# ----------------------------------------------kNN Method --------------------------------------------------------


def kNN(k, vector, metrics='euc'):
    if metrics == 'euc':
        allDistances = []
        for v in vectors:
            dist = EuclideanDistance(vector, v[1])
            allDistances.append((dist, v[0]))
        allDistances.sort(key=lambda y: y[0])
        counts = dict(Counter(elem[1] for elem in allDistances[:k]))
        label = max(counts, key=counts.get)
        return label
    elif metrics == 'cos':
        allSimilarities = []
        for v in vectors:
            similarity = CosinusSimilarity(vector, v[1])
            allSimilarities.append((similarity, v[0]))
        allSimilarities.sort(key=lambda y: y[0], reverse=True)
        counts = dict(Counter(elem[1] for elem in allSimilarities[:k]))
        label = max(counts, key=counts.get)
        return label


def TestkNN():
    k = 5
    p1 = 0
    p2 = 0
    p3 = 0
    p4 = 0

    print('kNN on train data with euclidean distance running...\n\n')
    # Euclidean Distance Learning error:
    file = open('optdigits.tra', 'r')
    lines = file.readlines()
    error = {i: 0 for i in range(0, 10)}
    processed = {i: 0 for i in range(0, 10)}

    for line in lines:
        print(lines.index(line))
        vector = np.fromstring(line, dtype=int, sep=',')
        label = vector[64]
        labelByEstimation = kNN(k, vector[:64], metrics='euc')
        if label != labelByEstimation:
            error[label] += 1
        processed[label] += 1
    print('kNN error percentage with euclidean distance on train data:')
    for i in range(0, 10):
        print('Number', i, ':', float((processed[i]-error[i]) / processed[i]) * 100)
        p1 += float((processed[i]-error[i]) / processed[i]) * 100
    print('\n#------------------------------------------------------------------------------------------#\n')

    # ------------------------------------------------------------------------------------------

    print('kNN on test data with euclidean distance running...\n\n')
    # Euclidean Distance Testing error:
    file = open('optdigits.tes', 'r')
    lines = file.readlines()
    error = {i: 0 for i in range(0, 10)}
    processed = {i: 0 for i in range(0, 10)}
    for line in lines:
        vector = np.fromstring(line, dtype=int, sep=',')
        label = vector[64]
        labelByEstimation = kNN(k, vector[:64], metrics='euc')
        if label != labelByEstimation:
            error[label] += 1
        processed[label] += 1
    print('kNN error percentage with euclidean distance on test data:')
    for i in range(0, 10):
        print('Number', i, ':', float((processed[i]-error[i]) / processed[i]) * 100)
        p2 += float(error[i] / processed[i]) * 100
    print('\n#------------------------------------------------------------------------------------------#\n')

    # ------------------------------------------------------------------------------------------

    print('kNN on train data with Cosinus Similarity running...\n\n')
    # Euclidean Distance Learning error:
    file = open('optdigits.tra', 'r')
    lines = file.readlines()
    error = {i: 0 for i in range(0, 10)}
    processed = {i: 0 for i in range(0, 10)}
    for line in lines:
        vector = np.fromstring(line, dtype=int, sep=',')
        label = vector[64]
        labelByEstimation = kNN(k, vector[:64], metrics='cos')
        if label != labelByEstimation:
            error[label] += 1
        processed[label] += 1
    print('kNN error percentage with Cosinus Similarity on train data:')
    for i in range(0, 10):
        print('Number', i, ':', float((processed[i]-error[i]) / processed[i]) * 100)
        p3 += float(error[i] / processed[i]) * 100
    print('\n#------------------------------------------------------------------------------------------#\n')

    # ------------------------------------------------------------------------------------------

    print('kNN on test data with Cosinus Similarity running...\n\n')
    # Euclidean Distance Testing error:
    file = open('optdigits.tes', 'r')
    lines = file.readlines()
    error = {i: 0 for i in range(0, 10)}
    processed = {i: 0 for i in range(0, 10)}
    for line in lines:
        vector = np.fromstring(line, dtype=int, sep=',')
        label = vector[64]
        labelByEstimation = kNN(k, vector[:64], metrics='cos')
        if label != labelByEstimation:
            error[label] += 1
        processed[label] += 1
    print('kNN error percentage with Cosinus Similarity on test data:')
    for i in range(0, 10):
        print('Number', i, ':', float((processed[i]-error[i]) / processed[i]) * 100)
        p4 += float(error[i] / processed[i]) * 100
    print('\n#------------------------------------------------------------------------------------------#\n')

    print('Overall: \n')
    print('Train data, Euclidean distance: ', p1 / 10, '\n')
    print('Test data, Euclidean distance: ', p2 / 10, '\n')
    print('Train data, Cosinus Similarity: ', p3 / 10, '\n')
    print('Train data, Cosinus Similarity: ', p4 / 10, '\n')


# ----------------------------------------------Centroid Method --------------------------------------------------------


def GetPrototypes():
    prototypes = []
    sortedVectors = sorted(vectors, key=lambda y: y[0])
    for i in range(10):
        prototype = np.zeros(64)
        counter = 0
        for vector in sortedVectors:
            if vector[0] > i:
                break
            elif vector[0] == i:
                counter += 1
                prototype += vector[1]
        prototype = np.divide(prototype, counter)
        prototypes.append(prototype)
    return prototypes


def CentroidMethod(vector, prototypes, metrics='euc'):
    if metrics == 'euc':
        distances = []
        for prototype in prototypes:
            distances.append(EuclideanDistance(vector, prototype))
        return distances.index(min(distances))
    else:
        allSimilarities = []
        for prototype in prototypes:
            allSimilarities.append(CosinusSimilarity(vector, prototype))
        return allSimilarities.index(max(allSimilarities))


def TestCentroidMethod():
    prototypes = GetPrototypes()
    p1 = 0
    p2 = 0
    p3 = 0
    p4 = 0

    print('Centroid method on train data with euclidean distance running...\n\n')
    # Euclidean Distance Learning error:
    file = open('optdigits.tra', 'r')
    lines = file.readlines()
    error = {i: 0 for i in range(0, 10)}
    processed = {i: 0 for i in range(0, 10)}
    for line in lines:
        vector = np.fromstring(line, dtype=int, sep=',')
        label = vector[64]
        labelByEstimation = CentroidMethod(vector[:64], prototypes)
        if label != labelByEstimation:
            error[label] += 1
        processed[label] += 1
    print('Centroid method error percentage with euclidean distance on train data:')
    for i in range(0, 10):
        print('Number', i, ':', float((processed[i]-error[i]) / processed[i]) * 100)
        p1 += float((processed[i]-error[i]) / processed[i]) * 100
    print('\n#------------------------------------------------------------------------------------------#\n')

    # ------------------------------------------------------------------------------------------

    print('kNN on test data with euclidean distance running...\n\n')
    # Euclidean Distance Testing error:
    file = open('optdigits.tes', 'r')
    lines = file.readlines()
    error = {i: 0 for i in range(0, 10)}
    processed = {i: 0 for i in range(0, 10)}
    for line in lines:
        vector = np.fromstring(line, dtype=int, sep=',')
        label = vector[64]
        labelByEstimation = CentroidMethod(vector[:64], prototypes)
        if label != labelByEstimation:
            error[label] += 1
        processed[label] += 1
    print('kNN error percentage with euclidean distance on test data:')
    for i in range(0, 10):
        print('Number', i, ':', float((processed[i]-error[i]) / processed[i]) * 100)
        p2 += float((processed[i]-error[i]) / processed[i]) * 100
    print('\n#------------------------------------------------------------------------------------------#\n')

    print('kNN on train data with Cosinus Similarity running...\n\n')
    # Euclidean Distance Learning error:
    file = open('optdigits.tra', 'r')
    lines = file.readlines()
    error = {i: 0 for i in range(0, 10)}
    processed = {i: 0 for i in range(0, 10)}
    for line in lines:
        vector = np.fromstring(line, dtype=int, sep=',')
        label = vector[64]
        labelByEstimation = CentroidMethod(vector[:64], prototypes, metrics='cos')
        if label != labelByEstimation:
            error[label] += 1
        processed[label] += 1
    print('kNN error percentage with Cosinus Similarity on train data:')
    for i in range(0, 10):
        print('Number', i, ':', float((processed[i]-error[i]) / processed[i]) * 100)
        p3 += float((processed[i]-error[i]) / processed[i]) * 100
    print('\n#------------------------------------------------------------------------------------------#\n')

    # ------------------------------------------------------------------------------------------

    print('kNN on test data with Cosinus Similarity running...\n\n')
    # Euclidean Distance Testing error:
    file = open('optdigits.tes', 'r')
    lines = file.readlines()
    error = {i: 0 for i in range(0, 10)}
    processed = {i: 0 for i in range(0, 10)}
    for line in lines:
        vector = np.fromstring(line, dtype=int, sep=',')
        label = vector[64]
        labelByEstimation = CentroidMethod(vector[:64], prototypes, metrics='cos')
        if label != labelByEstimation:
            error[label] += 1
        processed[label] += 1
    print('kNN error percentage with Cosinus Similarity on test data:')
    for i in range(0, 10):
        print('Number', i, ':', float((processed[i]-error[i]) / processed[i]) * 100)
        p4 += float((processed[i]-error[i]) / processed[i]) * 100
    print('\n#------------------------------------------------------------------------------------------#\n')

    print('Overall: \n')
    print('Train data, Euclidean distance: ', p1 / 10, '\n')
    print('Test data, Euclidean distance: ', p2 / 10, '\n')
    print('Train data, Cosinus Similarity: ', p3 / 10, '\n')
    print('Train data, Cosinus Similarity: ', p4 / 10, '\n')


# ----------------------------------------------Gradient descent -------------------------------------------------------

def GetWVal(label1, label2):
    X = []
    y = []
    gamma = 0.000035
    dataSize = 0
    for vector in vectors:
        if vector[0] == label1:
            X.append(np.append(vector[1], 1))
            y.append(+1)
            dataSize += 1
        elif vector[0] == label2:
            X.append(np.append(vector[1], 1))
            y.append(-1)
            dataSize += 1

    itr = 0
    w0 = np.zeros(65)
    while itr < 1800:
        itr += 1
        w0 = w0 - 2 * gamma / dataSize * np.matmul(np.transpose(X), (np.matmul(X, w0) - y))
    print(dataSize)
    return w0


def TestGradient(label1, label2):
    w = GetWVal(label1, label2)
    b = w[64]
    w = w[:64]
    processed = 0
    error = 0
    for vector in vectors:
        if vector[0] == label1 or vector[0] == label2:
            processed += 1
            pic = vector[1][:64]
            estimatedLabel = np.sign(np.matmul(np.transpose(w), pic) + b)
            if estimatedLabel < 0:
                estimatedLabel = label2
            else:
                estimatedLabel = label1
            if estimatedLabel != vector[0]:
                error += 1
    print('Train data error percentage: ', (processed - error) / processed * 100)

    processed = 0
    error = 0
    file = open('optdigits.tes', 'r')
    lines = file.readlines()
    for line in lines:
        inputVector = np.fromstring(line, dtype=int, sep=',')
        vector = (inputVector[64], inputVector[:64])
        if vector[0] == label1 or vector[0] == label2:
            processed += 1
            pic = vector[1][:64]
            estimatedLabel = np.sign(np.matmul(np.transpose(w), pic) + b)
            if estimatedLabel < 0:
                estimatedLabel = label2
            else:
                estimatedLabel = label1
            if estimatedLabel != vector[0]:
                error += 1
    print(error, processed)
    print('Test data error percentage: ', (processed - error) / processed * 100)


if __name__ == '__main__':
    ReadData()

    while True:
        cmd = input('Enter a method: ')
        if cmd == 'exit':
            exit(0)

        if cmd == 'knn':
            TestkNN()

        if cmd == 'centroid':
            TestCentroidMethod()

        if cmd == 'gradient':
            while True:
                a = input('Enter a number (0-9): ')
                if a == 'exit':
                    break
                a =int(a)
                b = int(input('Enter a number (0-9): '))
                TestGradient(a, b)
