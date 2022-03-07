import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

###################################
# PART A
###################################
A = np.array([[1, 2, 1, -1], [-1, 1, 0, 2], [0, -1, -2, 1]])
b = np.array([[3], [2], [-2]])

ALPHA = 0.1
GAMMA = 0.2

def funcf(x, gamma = GAMMA, alpha = ALPHA):
    return 1/2 * np.linalg.norm(A*x-b) + gamma/2*np.linalg.norm(x)

def gradFuncF(x, gamma = GAMMA):
    bT = np.transpose(b)
    AT = np.transpose(A)
    xT = np.transpose(x)
    Ax = np.dot(A, x)
    
    # working out on ipad
    return np.dot(AT, Ax-b) + gamma*x

oldX = np.array([[1], [1], [1], [1]])
grad = gradFuncF(oldX)
print(np.shape(grad))
print(np.shape(oldX))
print(grad)
k = 0
while np.linalg.norm(grad) >= 0.001:
    grad = gradFuncF(oldX)
    newX = oldX - ALPHA*grad
    print(f"k = {k} | x = {newX} | grad = {grad}")
    oldX = newX
    k += 1
locallyOptimalX = newX

###################################
# PART C
###################################
AT = np.transpose(A)
ATb = np.dot(AT, b)

bestX = np.dot(np.linalg.inv(np.dot(AT, A) + GAMMA * np.identity(4)), ATb)

print(bestX)

print(locallyOptimalX)

bestX - locallyOptimalX

###################################
# PART D
###################################

alphas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.02, 0.1, 0.15]

gamma = 0.2
maxK = 10000

def calcDifference(alpha):
    diffArray = []
    oldX = np.array([[1], [1], [1], [1]])
    grad = gradFuncF(oldX)
    k = 0
    while np.linalg.norm(grad) >= 0.001 and k < maxK:
        if k % 1000 == 0:
            print(f"alpha: {alpha} | k: {k} | norm of grad: {np.linalg.norm(grad)}")
        grad = gradFuncF(oldX)
        diffArray.append(np.linalg.norm(bestX - oldX))
        newX = oldX - alpha*grad
        oldX = newX
        k += 1
    return diffArray

alphaDiffs = []
for alpha in alphas:
    alphaDiffs.append(calcDifference(alpha))

# plots for D
for index, diffs in enumerate(alphaDiffs):
    x = list(range(0, len(diffs)))
    plt.title("L2 Norm Difference for all alphas")
    plt.plot(x, diffs, label = str(alphas[index]))
plt.legend()
plt.show()

axisMax = alphaDiffs[0][0] + 0.3 # The default Loss will be the largest + some buffer
axisMin = 0
for index, diffs in enumerate(alphaDiffs):
    x = list(range(0, len(diffs)))
    plt.title("L2 Norm Difference for alpha="+str(alphas[index]))
    plt.plot(x, diffs)
    plt.plot(x, [0.001], color="red")
    plt.ylim([axisMin, axisMax])
    plt.xlim([0, 10000])
    plt.show()

###################################
# PART E
###################################
rawdata = pd.read_csv('CarSeats.csv')
# remove the categorical varibales shelveLoc, urban, US
data = rawdata.drop(['ShelveLoc', 'Urban', 'US'], axis = 1)

# Scaling the data to normalize it
scaler = StandardScaler()
scaler.fit(data)
scaledData = scaler.transform(data)
scaledData = pd.DataFrame(scaledData)
scaledData.columns = (data.columns)
scaledData

scaledData['Sales'] = scaledData['Sales'] - scaledData['Sales'].mean()

print(scaledData.mean())
print(scaledData.var())

train = scaledData.iloc[:int(len(scaledData)/2), :]
test = scaledData.iloc[int(len(scaledData)/2):, :]

xTrain = train.drop(['Sales'], axis=1)
yTrain = train['Sales']
xTest = test.drop(['Sales'], axis=1)
yTest = test['Sales']

print(xTrain.head())
print(xTrain.tail())

print(yTrain.head())
print(yTrain.tail())

print(xTest.head())
print(xTest.tail())

print(yTest.head())
print(yTest.tail())

###################################
# PART G
###################################
x = xTrain
xT = np.transpose(x)
y = yTrain
I = np.identity(len(x.columns)) # x.columns is the parameters B
n = len(x)

bestParameters = 2/n * np.dot(np.linalg.inv(2/n * np.dot(xT, x) + I),np.dot(xT, y))
print(f"Answer to G: {bestParameters}")

###################################
# PART I
###################################
def calculateLoss(y, x, beta, theta, n):
    return 1/n * np.linalg.norm(y-np.dot(x, beta))+theta*np.linalg.norm(beta)

def gradFuncRidge(y, x, beta):
    x = np.asarray(x).reshape(x.shape[0], 1) # This is annoying I have to hardcode this in
    xT = np.transpose(x)
    return -2*np.dot(x, y - np.dot(xT, beta)) + 2*beta

def batchGD(yTrain, xTrain, alpha):
    oldBeta = np.array([[1], [1], [1], [1], [1], [1], [1]])
    bestBeta = oldBeta
    theta = 0.5
    n = len(xTrain)
    closedFormBestSolution = bestParameters
    diffArray = []
    epochLimit = 1000
    
    
    # Calculate the starting loss
    lossNewBeta = calculateLoss(np.asarray(yTrain), np.asarray(xTrain), oldBeta, theta, n)
    lossClosedForm = calculateLoss(yTrain, xTrain, closedFormBestSolution, theta, n)
    minDiff = lossNewBeta - lossClosedForm # starting difference
    bestBeta = oldBeta
    diffArray.append(minDiff)
    
    for epoch in range(0, epochLimit):
        gradSum = 0
        for i in range(0, n):
            # for each row
            gradSum += gradFuncRidge(yTrain[i], xTrain.iloc[i, :], oldBeta)
            
        newBeta = oldBeta - alpha / n * gradSum
        oldBeta = newBeta
        lossNewBeta = calculateLoss(np.asarray(yTrain), np.asarray(xTrain), newBeta, theta, n)
        lossClosedForm = calculateLoss(yTrain, xTrain, closedFormBestSolution, theta, n)
        diff = lossNewBeta - lossClosedForm
        if diff < minDiff:
            minDiff = diff
            bestBeta = newBeta
        diffArray.append(diff)
        if epoch % 333 == 0:
            print(f"alpha = {alpha} | epoch = {epoch} | diff = {diff}")
    
    return (diffArray, bestBeta)

alphasI = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

alphasDiffArrayI = []
alphasBetaArrayI = []

for alpha in alphasI:
    print("NEW ALPHA!")
    batchGDResult = batchGD(yTrain, xTrain, alpha)
    alphasDiffArrayI.append(batchGDResult[0])
    alphasBetaArrayI.append(batchGDResult[1])

# plots for i
for index, diffs in enumerate(alphasDiffArrayI):
    x = list(range(0, len(diffs)))
    plt.title("I: Difference for all alphas")
    plt.plot(x, diffs, label = str(alphasI[index]))
plt.legend()
plt.show()
    
axisMin = 0.5
axisMax = 4
for index, diffs in enumerate(alphasDiffArrayI):
    x = list(range(0, len(diffs)))
    plt.title("I: Difference for alpha="+str(alphasI[index]))
    plt.plot(x, diffs)
    plt.ylim([axisMin, axisMax])
    plt.xlim([0, 1000])
    plt.show()
    
# best alpha is = 0.15

# MSE I
nTrain = len(yTrain)
# beta for alpha = 0.01
bestTrainedBeta = batchGD(yTrain, xTrain, 0.01)[1]

batchGDTrainMSE = 1/n * np.linalg.norm(np.asarray(yTrain) - np.dot(np.asarray(xTrain), bestTrainedBeta))
batchGDTestMSE = 1/n * np.linalg.norm(np.asarray(yTest) - np.dot(np.asarray(xTest), bestTrainedBeta))

print(batchGDTrainMSE)
print(batchGDTestMSE)

###################################
# PART J
###################################

def calculateLoss(y, x, beta, theta, n):
    return 1/n * np.linalg.norm(y-np.dot(x, beta))+theta*np.linalg.norm(beta)

def gradFuncRidge(y, x, beta):
    x = np.asarray(x).reshape(x.shape[0], 1) # This is annoying I have to hardcode this in
    xT = np.transpose(x)
    return -2*np.dot(x, y - np.dot(xT, beta)) + 2*beta

def SGD(yTrain, xTrain, alpha):
    oldBeta = np.array([[1], [1], [1], [1], [1], [1], [1]])
    bestBeta = oldBeta
    
    closedFormBestSolution = bestParameters
    diffArray = []
    theta = 0.5
    epochLimit = 5
    n = len(xTrain)
    
    # Calculate the starting loss
    lossNewBeta = calculateLoss(np.asarray(yTrain), np.asarray(xTrain), oldBeta, theta, n)
    lossClosedForm = calculateLoss(yTrain, xTrain, closedFormBestSolution, theta, n)
    minDiff = lossNewBeta - lossClosedForm # starting difference
    bestBeta = oldBeta
    diffArray.append(minDiff)
    
    for epoch in range(0, epochLimit):
        for i in range(0, n):
            # for each row
            grad = gradFuncRidge(yTrain[i], xTrain.iloc[i, :], oldBeta)
            newBeta = oldBeta - alpha * grad
            oldBeta = newBeta
            
            # The loss calculation (done each time beta is updated)
            lossNewBeta = calculateLoss(np.asarray(yTrain), np.asarray(xTrain), newBeta, theta, n)
            lossClosedForm = calculateLoss(yTrain, xTrain, closedFormBestSolution, theta, n)
            diff = lossNewBeta - lossClosedForm
            if diff < minDiff:
                minDiff = diff
                bestBeta = newBeta
            diffArray.append(diff)
        
        if epoch % 333 == 0:
            print(f"alpha = {alpha} | epoch = {epoch} | diff = {diff}")
    
    return (diffArray, bestBeta)

alphasJ = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02]

alphasDiffArrayJ = []
alphasBetaArrayJ = []

for alpha in alphasJ:
    print("NEW ALPHA!")
    SGDResult = SGD(yTrain, xTrain, alpha)
    alphasDiffArrayJ.append(SGDResult[0])
    alphasBetaArrayJ.append(SGDResult[1])

# plots for J
for index, diffs in enumerate(alphasDiffArrayJ):
    x = list(range(0, len(diffs)))
    plt.title("J: Difference for all alphas")
    plt.plot(x, diffs, label = str(alphasJ[index]))
plt.legend()
plt.show()
    
for index, diffs in enumerate(alphasDiffArrayJ):
    x = list(range(0, len(diffs)))
    plt.title("J: Difference for alpha="+str(alphasJ[index]))
    plt.plot(x, diffs)
    plt.show()

# best alpha is 0.02

# MSE J
nTrain = len(yTrain)
# beta for alpha = 0.01
bestTrainedBetaJ = SGD(yTrain, xTrain, 0.02)[1]

SGDTrainMSE = 1/n * np.linalg.norm(np.asarray(yTrain) - np.dot(np.asarray(xTrain), bestTrainedBetaJ))
SGDTestMSE = 1/n * np.linalg.norm(np.asarray(yTest) - np.dot(np.asarray(xTest), bestTrainedBetaJ))

print(SGDTrainMSE)
print(SGDTestMSE)