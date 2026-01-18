import numpy as np
import math

def L1Distance(vector_1, vector_2):

    L1 = 0.0
    for e1, e2 in zip(vector_1, vector_2):
        L1 += abs(e1 - e2)
    return L1

def L2Distance(vector_1, vector_2):

    L2 = 0.0
    for e1, e2 in zip(vector_1, vector_2):
        L2 += (e1 - e2) ** 2
    L2 = math.sqrt(L2)
    return L2

def Cosine(vector_1, vector_2):
    sum = np.dot(np.mat(vector_1),  np.mat(vector_2).T)

    denom = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    return sum/denom


def VectorKLDivergence(vector1, vector2):

    if vector1.shape != vector2.shape:
        print('the shape between two vectors is different.')
        return 0.0
    
    p1 = 1.0 *vector1 /np.sum(vector1)
    p2 = 1.0 *vector2 /np.sum(vector2)

    return np.sum(np.multiply(p1, (np.log(p1) - np.log(p2))))

def MatrixKLDivergence_SameShape(matrix1, matrix2):

    if matrix1.shape[1] != matrix2.shape[1]:
        print('the second dimension of shape between two matrices is different.')
        return 0.0

    p1 = 1.0 * matrix1 / np.sum(matrix1, axis=1)[:,None]
    p2 = 1.0 * matrix2 / np.sum(matrix2, axis=1)[:,None]

    return np.sum(np.multiply(p1, (np.log(p1) - np.log(p2))), axis=1)


def MatrixKLDivergence_SameSecondDimension(matrix1, matrix2):

    if matrix1.shape[1] != matrix2.shape[1]:
        print('the second dimension of shape between two matrices is different.')
        return 0.0

    p1 = 1.0 * matrix1 / np.sum(matrix1, axis=1)[:,None]
    log1 = np.log(p1)
    p2 = 1.0 * matrix2 / np.sum(matrix2, axis=1)[:,None]
    log2 = np.log(p2)

    log2Extension = np.empty(shape=(0, matrix1.shape[1]))
    for row in log2:

        temp = log1 - (np.tile(row, (matrix1.shape[0], 1)))
        log2Extension = np.vstack((log2Extension, temp))

    log2Extension = log2Extension.reshape(matrix2.shape[0], matrix1.shape[0], matrix1.shape[1])

    KLDivergence = np.sum(np.multiply(p1, log2Extension), axis=2)
    return KLDivergence.T
