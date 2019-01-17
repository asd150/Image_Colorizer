import numpy as np
import cv2
import random
import os

def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

def getNeighbors(matrix, x, y):

    arrayNeighbors = [ [], [], [], [], [], [], [], [], [] ]
    arrayNeighbors[0] = [matrix[x - 1][y - 1]]
    arrayNeighbors[1] = [matrix[x - 1][y]]
    arrayNeighbors[2] = [matrix[x - 1][y + 1]]
    arrayNeighbors[3] = [matrix[x][y - 1]]
    arrayNeighbors[4] = [matrix[x][y]]
    arrayNeighbors[5] = [matrix[x][y + 1]]
    arrayNeighbors[6] = [matrix[x + 1][y - 1]]
    arrayNeighbors[7] = [matrix[x + 1][y]]
    arrayNeighbors[8] = [matrix[x + 1][y + 1]]

    return np.array(arrayNeighbors)

def duplicateEdges(grayMatrix):
    grayMatrixPadded = np.pad(grayMatrix, pad_width=1, mode='constant', constant_values=0)

    for r in range(len(grayMatrixPadded) - 1):
        if ((not (r == 0)) and (not (r == (len(grayMatrixPadded) - 1)))):
            grayMatrixPadded[r][0] = grayMatrixPadded[r][1]

    for s in range(len(grayMatrixPadded) - 1):
        if ((not (s == 0)) and (not (s == (len(grayMatrixPadded) - 1)))):
            grayMatrixPadded[s][(len(grayMatrixPadded[0]) - 1)] = grayMatrixPadded[s][(len(grayMatrixPadded[0]) - 2)]

    for t in range(len(grayMatrixPadded[0]) - 1):
        if ((not (t == 0)) and (not (t == (len(grayMatrixPadded[0]) - 1)))):
            grayMatrixPadded[0][t] = grayMatrixPadded[1][t]

    for b in range(len(grayMatrixPadded[0]) - 1):
        if ((not (b == 0)) and (not (b == (len(grayMatrixPadded[0]) - 1)))):
            grayMatrixPadded[(len(grayMatrixPadded) - 1)][b] = grayMatrixPadded[(len(grayMatrixPadded) - 2)][b]

    grayMatrixPadded[0][0] = 128
    grayMatrixPadded[0][(len(grayMatrixPadded[0]) - 1)] = 128
    grayMatrixPadded[(len(grayMatrixPadded) - 1)][0] = 128
    grayMatrixPadded[(len(grayMatrixPadded) - 1)][(len(grayMatrixPadded[0]) - 1)] = 128

    return np.array(grayMatrixPadded)

def colorImage(directoryOfTrainingDataWeights, directoryOfGrayImg):
    weights_ONE = np.loadtxt(str(directoryOfTrainingDataWeights)+"/weights_one.txt")
    weights_TWO = np.loadtxt(str(directoryOfTrainingDataWeights)+"/weights_two.txt")
    weights_THREE = np.loadtxt(str(directoryOfTrainingDataWeights)+"/weights_three.txt")

    img_file = str(directoryOfGrayImg)
    gray_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)  # grayscale
    cv2.imwrite("grayScaled.jpg", gray_img)
    img = cv2.imread("grayScaled.jpg", 0)

    height = np.size(img, 0)
    width = np.size(img, 1)

    grayMatrix = [[0 for x in range(width)] for y in range(height)]
    for i in range(height):  # traverses through height of the image
        for j in range(width):  # traverses through width of the image
            grayMatrix[i][j] = img[i][j]

    grayMatrix = np.array(grayMatrix)
    # grayMatrixPadded = duplicateEdges(grayMatrix)


    rows = len(grayMatrix)
    cols = len(grayMatrix[0])

    B_output = np.zeros((rows, cols), dtype='uint8')
    G_output = np.zeros((rows, cols), dtype='uint8')
    R_output = np.zeros((rows, cols), dtype='uint8')

    print("Testing random image: ")

    for i in range(1, len(grayMatrix) - 1):
        for j in range(1, len(grayMatrix[0]) - 1):
            window = getNeighbors(grayMatrix, i, j) / 255  # gets a 9x1 matrix of neighbors
            layer1 = sigmoid(np.dot(weights_ONE, window))  # 4x1
            layer2 = sigmoid(np.dot(weights_TWO, layer1))  # 3x1
            output = sigmoid(np.dot(weights_THREE, layer2))

            R_output[i][j] = output[0][0] * 255
            G_output[i][j] = output[1][0] * 255
            B_output[i][j] = output[2][0] * 255

    mergedChannels = cv2.merge((B_output, G_output, R_output))
    cv2.imshow("output", mergedChannels)
    cv2.waitKey(0)

def trainNeuralNetwork(pathOfTrainingData, desiredDirectoryToStoreWeights):

    np.random.seed(1)
    weights_ONE = 2 * np.random.random((255, 9)) - 1
    weights_TWO = 2 * np.random.random((255, 255)) - 1
    weights_THREE = 2 * np.random.random((3, 255)) - 1

    for filename in os.listdir(pathOfTrainingData):
        print("Training file: ", filename)
        img_file = str(pathOfTrainingData + filename)

        gray_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)  # grayscale
        cv2.imwrite("grayScaled.jpg", gray_img)
        img = cv2.imread("grayScaled.jpg", 0)

        height = np.size(img, 0)
        width = np.size(img, 1)

        grayMatrix = [[0 for x in range(width)] for y in range(height)]
        for i in range(height):  # traverses through height of the image
            for j in range(width):  # traverses through width of the image
                grayMatrix[i][j] = img[i][j]

        grayMatrix = np.array(grayMatrix)
        # grayMatrixPadded = duplicateEdges(grayMatrix)

        color_img = cv2.imread(img_file)
        blue, green, red = cv2.split(color_img)

        blue = np.array(blue)
        green = np.array(green)
        red = np.array(red)

        for x in range(1):
            # if x%len(grayMatrix) == 0:
            #     print(len(grayMatrix)/x)
            # for i in range(1, len(grayMatrix) - 1):
            #     for j in range(1, len(grayMatrix[0]) - 1):
            # for i in range(len(grayMatrix)-2, 1, -1):
            #     for j in range(len(grayMatrix[0])-2, 1, -1):

            num_of_rows = len(grayMatrix)
            num_of_cols = len(grayMatrix[0])
            tuples = []

            for i in range(1, num_of_rows - 1):
                for j in range(1, num_of_cols - 1):
                    tuples.append((i, j))

            while len(tuples) >= 1:
                rand_num = random.randint(0, len(tuples) - 1)
                curr_tuple = tuples.pop(rand_num)
                i = curr_tuple[0]
                j = curr_tuple[1]
                # print(len(tuples))
                window = getNeighbors(grayMatrix, i, j) / 255  # gets a 9x1 matrix of neighbors
                center_rgb_pixel = np.array([[red[i][j]], [green[i][j]], [blue[i][j]]]) / 255  # a 3x1 matrix of rgb

                layer1 = sigmoid(np.dot(weights_ONE, window))  # 4x1
                layer2 = sigmoid(np.dot(weights_TWO, layer1))
                output = sigmoid(np.dot(weights_THREE, layer2))  # 3x1
                error = center_rgb_pixel - output  # 3x1 - 3x1

                l3_delta = error * sigmoid(output, True)  # 3x1
                l2_error = weights_THREE.T.dot(l3_delta)
                l2_delta = l2_error * sigmoid(layer2, True)  # 3x1

                l1_error = weights_TWO.T.dot(l2_delta)  # 4x1
                l1_delta = l1_error * sigmoid(layer1, True)  # 4x1

                weights_THREE += l3_delta.dot(layer2.T)
                weights_TWO += l2_delta.dot(layer1.T)  # 3x4
                weights_ONE += l1_delta.dot(window.T)  # 4x9

            print("Error: " + str(np.mean(np.abs(error))))


    if not os.path.exists(desiredDirectoryToStoreWeights):
        os.makedirs(desiredDirectoryToStoreWeights)
        np.savetxt(desiredDirectoryToStoreWeights+"/weights_one.txt", weights_ONE)
        np.savetxt(desiredDirectoryToStoreWeights+"/weights_two.txt", weights_TWO)
        np.savetxt(desiredDirectoryToStoreWeights+"/weights_three.txt", weights_THREE)
    else:
        raise ValueError('The directory already exists. Weights were not saved. Remove directory or change desiredDirectoryFolderNameToStoreWeights')


if __name__ == '__main__':

    # Specify absolute path of main directory of 'Colorizer'

    mainPath = "/Users/chiragchadha/OneDrive/Colorizer"

    # Specify Relative Path of Training Data Images
    relativePathTrainingData = "Train/Earth"

    # Create a name of desired directory. Do not create directory. Program will handle automatically. Make sure named direcory does not currently exist
    desiredDirectoryFolderNameToStoreWeights = "earth_weights"

    #Specify absolute path of image you would like to color using neural network
    pathOfGrayImg = "/Users/chiragchadha/OneDrive/Colorizer/Colorize/earth_test.jpg"

    #------------------------------------------------- LEAVE AS IS -------------------------------------------------------------#
    mainPath = mainPath + "/"
    trainingDataDirectory = mainPath + relativePathTrainingData + "/"
    desiredDirectoryToStoreWeights = mainPath + desiredDirectoryFolderNameToStoreWeights
    #------------------------------------------------- LEAVE AS IS -------------------------------------------------------------#


    #Program will use directory that was recently created to store trained neural network. If you would like to use different set of weights, specify absolute path of weights.
    directoryOfStoredWeights = desiredDirectoryToStoreWeights + "/"

    #EXAMPLE OF ALTERNATIVE: directoryOfStoredWeights = "/Users/chiragchadha/OneDrive/Colorizer/beach_weights/"


    #If you would like to use pretrained models, comment out the trainNueralNetwork() function
    trainNeuralNetwork(trainingDataDirectory, desiredDirectoryToStoreWeights)

    colorImage(directoryOfStoredWeights, pathOfGrayImg)






