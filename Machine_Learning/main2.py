import random

def main():
  random.seed()
  #weights for each input
  weights = []
  n = 0.01
  train(weights, n, "mnist_train.csv", "mnist_test.csv", "accuracyTrain01.csv", "accuracyTest01.csv", "conMatrixTest01.csv")
  
def train(weights, n, inputFileTrain, inputFileTest, outputFileTrain, outputFileTest, conMatrixFile):
  epoch = 0
  numItemsTrain = 0
  numItemsTest = 0
  hitTrain = 0
  hitTest = 0
  accuracyTrain = 0.0
  prevAccuracyTrain = 0.0
  accuracyTest = 0.0
  prevAccuracyTest = 0.0
  #labels for what the value should be
  labelsTrain = []
  labelsTest = []
  #inputs for the image (0 is white, 255 is black)
  inputsTrain = []
  inputsTest = []
  #counter for perceptrons
  i = 0
  #counter to end early
  j = 0
  conMatrix = [[0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0]]

  for p in range(785):
    weights.append(randomWeights())
  
  perceptrons = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

  #TRAIN
  with open(inputFileTrain, 'r') as dataSet:
    for line in dataSet:
      #create input, labels
      inputsTrain.append(createInputTrain(line, labelsTrain))
      numItemsTrain += 1

  #TEST
  with open(inputFileTest, 'r') as dataSet:
    for line in dataSet:
      #create input, labels
      inputsTest.append(createInputTrain(line, labelsTest))
      numItemsTest += 1

  #TRAIN
  #set up starting data point at epoch 0
  d = 0
  for inputSet in inputsTrain:
    for x, weightSet in zip(inputSet, weights):
      for w in weightSet:
        perceptrons[i] += x * w
        i += 1
      i = 0
    if perceptrons.index(max(perceptrons)) == labelsTrain[d]:
      hitTrain += 1
    conMatrix[labelsTrain[d]][perceptrons.index(max(perceptrons))] += 1
    #zero perceptrons
    perceptrons = list(map(zero, perceptrons))
    d += 1
  accuracyTrain = (hitTrain/numItemsTrain) * 100
  with open(outputFileTrain, 'a') as f:
    f.write(str(epoch) + ", " + str(accuracyTrain) + "\n")
  epoch = 0
  hitTrain = 0

  #TEST
  #set up starting data point at epoch 0
  o = 0
  for inputSet in inputsTest:
    for x, weightSet in zip(inputSet, weights):
      for w in weightSet:
        perceptrons[i] += x * w
        i += 1
      i = 0
    if perceptrons.index(max(perceptrons)) == labelsTest[o]:
      hitTest += 1
    conMatrix[labelsTest[o]][perceptrons.index(max(perceptrons))] += 1
    #zero perceptrons
    perceptrons = list(map(zero, perceptrons))
    o += 1
  accuracyTest = (hitTest/numItemsTest) * 100
  with open(outputFileTest, 'a') as f:
    f.write(str(epoch) + ", " + str(accuracyTest) + "\n")
  epoch = 0
  hitTest = 0

  epoch += 1

  #TRAIN                   
  while accuracyTrain - prevAccuracyTrain > 0.01 and j < 70:
    #SUM EVEYRTHING
    print(str(j))
    k = 0
    for inputSet in inputsTrain:
      for x, weightSet in zip(inputSet, weights):
        for w in weightSet:
          perceptrons[i] += x * w
          i += 1
        i = 0
      #if prediction does not equal the class, change the weights
      if perceptrons.index(max(perceptrons)) != labelsTrain[k]:
        h = 0
        for x, weightSet in zip(inputSet, weights):
          newSubWL = []
          for w in weightSet:
            t = 0
            y = 0
            if perceptrons[i] > 0:
              y = 1
            if i == labelsTrain[k]:
              t = 1
            newW = w + (n * (t - y) * x)
            newSubWL.append(newW)
            i += 1
          i = 0
          weights[h] = newSubWL
          h += 1
      else:
        hitTrain += 1
      conMatrix[labelsTrain[k]][perceptrons.index(max(perceptrons))] += 1
      #ZERO
      perceptrons = list(map(zero, perceptrons))
      k += 1
    prevAccuracyTrain = accuracyTrain
    accuracyTrain = (hitTrain/numItemsTrain) * 100
    with open(outputFileTrain, 'a') as f:
      f.write(str(epoch) + ", " + str(accuracyTrain) + "\n")

    #TEST
    #set up starting data point at epoch 0
    o = 0
    for inputSet in inputsTest:
      for x, weightSet in zip(inputSet, weights):
        for w in weightSet:
          perceptrons[i] += x * w
          i += 1
        i = 0
      if perceptrons.index(max(perceptrons)) == labelsTest[o]:
        hitTest += 1
      conMatrix[labelsTest[o]][perceptrons.index(max(perceptrons))] += 1
      #zero perceptrons
      perceptrons = list(map(zero, perceptrons))
      o += 1
    accuracyTest = (hitTest/numItemsTest) * 100
    with open(outputFileTest, 'a') as f:
      f.write(str(epoch) + ", " + str(accuracyTest) + "\n")
    hitTest = 0
    epoch += 1
    j += 1
    hitTrain = 0
                        
  with open(conMatrixFile, 'w') as matrix:
    for line in conMatrix:
      line = str(line)
      line = line[:-1]
      line = line[1:]
      matrix.write(line + "\n")

def small(x):
  return x/255

def zero(x):
  return abs(x*0)

def createInputTrain(line, labels):
  i = 0
  line = line.split(',')
  line[-1] = line[-1].rstrip()
  line = list(map(int, line))
  labels.append(line.pop(0))
  line = list(map(float, line))
  line = list(map(small, line))
  #prepends bias input
  line.insert(0, 1.0)
  return line

def randomWeights():
  weights = []
  for i in range(0,10):
    weights.append(random.uniform(-0.5, 0.5))
  return weights
  
  
if __name__== "__main__":
  main()
