from __future__ import division
import pandas as pd
import math
import Tree
from time import time
import numpy


class Regression:
    def __init__(self, dataSet, fold, nmins):
        self.dataSet = dataSet
        self.nmins = nmins
        self.fold = fold
        

    def normalize(self):
        for attribute in self.attributes:
            minAtt = self.dataSet[attribute].min()
            maxAtt= self.dataSet[attribute].max()
            self.dataSet[attribute] = self.dataSet[attribute].apply(lambda x: (x - minAtt) / (maxAtt - minAtt))
   
    def split(self, data, attribute, value):
        result= []
        result.append(data.loc[data[attribute] <= value])
        result.append(data.loc[data[attribute] > value])
        return result
    
    
    
    
    def isPure(self, dataSet):
        return dataSet[self.targetAttribute].unique().shape[0] == 1

    def getAttval(self, data, attribute):
        Vals = set()
        sortedData = data.sort_values(attribute)
        previousVal = sortedData[attribute].iloc[0]
        for index, row in sortedData[1:].iterrows():
            currentVal = row[attribute]
            Vals.add((previousVal + currentVal) / 2)
            previousVal = currentVal
        return list(Vals )


    def ErrorSum(self, data):
        error = 0
        mean = data[self.targetAttribute].mean()
        for index, row in data.iterrows():
            error = error+ (row[self.targetAttribute] - mean)**2
        return error



    def lowerError(self, dataSet, sse, attribute, value):
        for child in self.split(dataSet, attribute, value):
            sse =sse- (child.shape[0] / dataSet.shape[0]) * self.ErrorSum(child)
        return sse

    def getLowerError(self, data, sse, attribute):
        vals = self.getAttval(data, attribute)
        value = vals[0]
        error = self.lowerError(data, sse, attribute, value)
        for newValue in vals[1:]:
            newError = self.lowerError(data, sse, attribute, newValue)
            if newError < error:
                error = newError
                value = newValue
        return error, value

    def getHighestAttribute(self, data, sse, attributes):
        Highest = attributes[0]
        error, attributeVal = self.getLowerError(data, sse, Highest)
        for attribute in attributes[1:]:
            newError, newAttributeVal = self.getLowerError(
                data, sse, attribute)
            if newError < error:
                error = newError
                attributeVal = newAttributeVal
                Highest = attribute
        return Highest, attributeVal

    def classify(self, decisionTree, data):
        if decisionTree.isLeaf:
            return decisionTree.label
        else:
            value = data[decisionTree.attribute]
            if value <= decisionTree.value:
                return self.classify(decisionTree.trueBranch, data)
            else:
                return self.classify(decisionTree.falseBranch, data)

    def predict(self, Tree, dataSet):
        actualVal = []
        predictedVal = []
        for index, row in dataSet.iterrows():
            actualVal.append(row[self.targetAttribute])
            predictedVal.append(self.classify(Tree, row))
        error = 0
        for i in range(len(actualVal)):
            error =error+ (actualVal[i] - predictedVal[i])**2
        error /= len(actualVal)
        return math.sqrt(error)
    
    def makeTree(self, dataSet, threshold, attributes):
        root = Tree.MultiSplitTree()
        mostCommonLabel = dataSet[self.targetAttribute].mean()
        if self.isPure(dataSet) or  dataSet.shape[0] < threshold or len(attributes) == 0 :
            root.isLeaf = True
            root.label = mostCommonLabel
            return root
        
        else:
            root.isTree = True
            root.mostCommonLabel = mostCommonLabel
            root.entropy = self.ErrorSum(dataSet)
            attribute, attributeValue = self.getHighestAttribute(dataSet, root.entropy, attributes)
            root.attribute = attribute
            newAtt = attributes[:]
            newAtt.remove(attribute)
            root.value = attributeValue
            
            subGroup = dataSet.loc[dataSet[attribute] <= attributeValue]
            
            if (subGroup.shape[0] > 0):
                treeBranch = self.makeTree(subGroup, threshold, newAtt)
                root.trueBranch = treeBranch
            else:
                treeBranch = Tree.MultiSplitTree()
                treeBranch.label = mostCommonLabel
                treeBranch.isLeaf = True
                root.trueBranch = treeBranch
                
            subGroup = dataSet.loc[dataSet[attribute] > attributeValue]
            
            if (subGroup.shape[0] > 0):
                treeBranch = self.makeTree(subGroup, threshold,newAtt)
                root.falseBranch = treeBranch
            else:
                treeBranch = Tree.MultiSplitTree()
                treeBranch.label = mostCommonLabel
                treeBranch.isLeaf = True
                root.falseBranch = treeBranch
        return root

  

    def validate(self):
        cols = list(self.dataSet)
        self.attributes = cols[0:len(cols) - 1]
        self.targetAttribute = cols[len(cols) - 1]
        self.normalize()
        
        start = 0
        end = 0
        rows = self.dataSet.shape[0]
        Range = rows // self.fold
        shuffledDataSet = self.dataSet.sample(frac=1)
        dataSets = []
   
        
        for current in range(self.fold):
            start =  current * Range
            if ( current + 2) * Range <= rows:
                end = ( current + 1) * Range
            else:
                end = rows
            dataSets.append(shuffledDataSet[start:end])
        for nmin in self.nmins:
            threshold = nmin * rows
            print('----------------------------')
            print()
            print("NMIN :: {}".format(nmin))
            TrainErrors = []
            TestErrors = []
            start = time()
            for i in range(self.fold):
                trainDataSet = pd.concat( dataSets[0:i] + dataSets[i + 1:self.fold])
                testDataSet = dataSets[i]
                print("CURRENT FOLD :: {}".format(i + 1))
                decisionTree = self.makeTree(trainDataSet, threshold,self.attributes)
                TrainError = self.predict(decisionTree, trainDataSet)
                TrainErrors.append(TrainError)
                print("TRAINING SET :: {} :: SSE :: {}".format(
                    trainDataSet.shape[0], TrainError))
                TestError = self.predict(decisionTree, testDataSet)
                TestErrors.append(TestError)
                print("TEST SET :: {} :: SSE :: {}".format( testDataSet.shape[0], TestError))
            end = time()
            trainSSE = numpy.mean(TrainErrors)
            testSSE = numpy.mean(TestErrors)
            testStandardDeviation = numpy.std(TestErrors)
            print('==========================')
            print()
            print(
                "TRAIN SSE :: {} :: TEST SSE :: {} :: STANDARD DEVIATION :: {}".
                format(trainSSE, testSSE, testStandardDeviation))
            print("TIME :: {}".format(end - start))
            print ('Regression ends')
            print ()