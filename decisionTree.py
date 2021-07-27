from __future__ import division
import numpy
import pandas as pd
from time import time
from pandas_ml import ConfusionMatrix
import Tree
import numpy as np



class decisionTree:
    def __init__(self, dataSet, fold, nmins, binary=False):
        
        self.dataSet = dataSet
        self.nmins = nmins
        self.binary= binary
        self.fold = fold

    def normalize(self):
        if self.binary:
            for attribute in self.attributes:
                minAtt = self.dataSet[attribute].min()
                maxAtt = self.dataSet[attribute].max()
                self.dataSet[attribute] = self.dataSet[attribute].apply(lambda x: (x - minAtt) / (maxAtt - minAtt))

    def split(self, data, attribute, value=None):
        result= []
        if self.binary:
            result.append(data.loc[data[attribute] < value])
            result.append(data.loc[data[attribute] >= value])
        else:
            for value in data[attribute].unique():
                result.append(data.loc[data[attribute] == value])
        return result
    
    def isPure(self, data):
        return data[self.targetAttribute].unique().shape[0] == 1


    def getEntropy(self, data):
        splits = data[self.targetAttribute].value_counts().to_dict()
        entropy = 0.0
        for label, c in splits.items():
            frequency = c / data.shape[0]
            entropy = entropy - frequency * np.log2(frequency)
        return entropy


    def getAttributeVal(self, data, attribute):
        vals = set()
        sortedData = data.sort_values(attribute)
        value = sortedData[attribute].iloc[0]
        label = sortedData[self.targetAttribute].iloc[0]
        for index, row in sortedData[1:].iterrows():
            newLabel = row[self.targetAttribute]
            newValue = row[attribute]
            if newLabel != label:
                vals.add((value + newValue)/2)
                label = newLabel
                value = newValue
        return list(vals)

    def getInformationGain(self, data, entropy, attribute, value=None):
        InformationGain = entropy
        for dataset in self.split(data, attribute, value):
            InformationGain = InformationGain - (dataset.shape[0] / data.shape[0]) * self.getEntropy(dataset)
        return InformationGain

    def informationGainProcess(self, dataSet, entropy, attribute):
        if self.binary:
            attributeValues = self.getAttributeVal(dataSet, attribute)
            attributeValue = attributeValues[0]
            informationGain = self.getInformationGain(
                dataSet, entropy, attribute, attributeValue)
            for newAttributeValue in attributeValues[1:]:
                newInformationGain = self.getInformationGain(
                    dataSet, entropy, attribute, newAttributeValue)
                if newInformationGain > informationGain:
                    informationGain = newInformationGain
                    attributeValue = newAttributeValue
            return informationGain, attributeValue
        else:
            return self.getInformationGain(dataSet, entropy, attribute,
                                           None), 0.0

    def getHighestAttribute(self, data, entropy, attributes):
        highest = attributes[0]
        informationGain, attributeValue = self.informationGainProcess(
            data, entropy, highest )
        for attribute in attributes[1:]:
            newInformationGain, newAttributeValue = self.informationGainProcess(
                data, entropy, attribute)
            if newInformationGain > informationGain:
                informationGain = newInformationGain
                attributeValue = newAttributeValue
                highest  = attribute
        return highest, attributeValue

    def classify(self, decisionTree, data):
        if decisionTree.isLeaf:
            return decisionTree.label
        else:
            value = data[decisionTree.attribute]
            if self.binary:
                if value >= decisionTree.value:
                    
                    return self.classify(decisionTree.falseChildren, data)
                else:
                    return self.classify(decisionTree.trueChildren, data)
                   
            else:
                Children = filter(lambda x: x.value == value,decisionTree.children)
               
                Children=(list(Children))
                length=len(Children)
                if length == 0:
                    return decisionTree.mostLabel
                else:
                    return self.classify(Children[0], data)

    def predictResult(self, decisionTree, dataSet, test=False):
        error = 0
        for index, row in dataSet.iterrows():
            actualResult = row[self.targetAttribute]
            prediction = self.classify(decisionTree, row)
            if test:
                self.actualValues.append(actualResult)
                self.predictedValues.append(prediction)
            if prediction != actualResult:
                error += 1
        return error

    def calculateConfusionMatrixStats(self):
        confusion_matrix = ConfusionMatrix(self.actualValues,self.predictedValues)
        confusion_matrix.print_stats()



    def makeTree(self, dataSet, threshold, attributes):
        root = Tree.MultiSplitTree()
        mostLabel = dataSet[self.targetAttribute].value_counts().idxmax()
        if self.isPure(dataSet) or len(
                attributes) == 0 or dataSet.shape[0] < threshold:
            root.isLeaf = True
            root.label = mostLabel
            
            return root
        else:
            root.isTree = True
            root.mostLabel = mostLabel
            root.entropy = self.getEntropy(dataSet)
            
            attribute, attributeValue = self.getHighestAttribute(dataSet, root.entropy, attributes)
            root.attribute = attribute
            newAtt = attributes[:]
            newAtt.remove(attribute)
            if self.binary:
                root.value = attributeValue
                subSet = dataSet.loc[dataSet[attribute] < attributeValue]
                if (subSet.shape[0] > 0):
                    treeBranch = self.makeTree(subSet, threshold, newAtt)
                    root.trueChildren = treeBranch
                else:
                    
                    treeBranch = Tree.MultiSplitTree()
                    treeBranch.isLeaf = True
                    root.trueChildren = treeBranch
                    treeBranch.label = mostLabel
                    
                subSet = dataSet.loc[dataSet[attribute] >= attributeValue]
                
                if (subSet.shape[0] > 0):
                    treeBranch = self.makeTree(subSet, threshold, newAtt)
                    root.falseChildren= treeBranch
                else:
                    treeBranch = Tree.MultiSplitTree()
                    
                    treeBranch.isLeaf = True
                    root.falseChildren= treeBranch
                    treeBranch.label = mostLabel
            else:
                for value in dataSet[attribute].unique():
                    subSet = dataSet.loc[dataSet[attribute] == value]
                    treeBranch = self.makeTree(subSet, threshold, newAtt)
                    treeBranch.value = value
                    root.children.append(treeBranch)
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
            start = current *  Range 
            if (current + 2) *  Range  <= rows:
                end = (current + 1) *  Range 
            else:
                end = rows
            dataSets.append(shuffledDataSet[start:end])
            
        for nmin in self.nmins:
            threshold = nmin * rows
            print('----------------------------')
            print()
            print("NMIN is :: {}".format(nmin))
            setTrainAccuracies = []
            setTestAccuracies = []
            self.actualValues = []
            self.predictedValues = []
            start = time()
            for i in range(self.fold):
                trainDataSet = pd.concat(dataSets[0:i] + dataSets[i + 1:self.fold])
                testDataSet = dataSets[i]
                print("CURRENT FOLD :: {}".format(i + 1))
                decisionTree = self.makeTree(trainDataSet, threshold,self.attributes)
                setTrainError = self.predictResult(decisionTree, trainDataSet, False)
                setTrainAccuracy = (trainDataSet.shape[0] - setTrainError) * 100 / trainDataSet.shape[0]
                setTrainAccuracies.append(setTrainAccuracy)
                print("TRAINING  :: {} :: ERROR :: {} :: ACCURACY :: {}".
                      format(trainDataSet.shape[0], setTrainError,setTrainAccuracy))
                setTestError = self.predictResult(decisionTree, testDataSet, True)
                setTestAccuracy = (testDataSet.shape[0] - setTestError) * 100 / testDataSet.shape[0]
                setTestAccuracies.append(setTestAccuracy)
                print("TEST :: {} :: ERROR :: {} :: ACCURACY :: {}".format(
                    testDataSet.shape[0], setTestError, setTestAccuracy))
            end = time()
            trainAccuracy = numpy.mean(setTrainAccuracies)
            testAccuracy = numpy.mean(setTestAccuracies)
            testStandardDeviation = numpy.std(setTestAccuracies)
            print('==========================')
            print()
            print(
                "TRAIN ACCURACY :: {} :: TEST ACCURACY :: {} :: STANDARD DEVIATION :: {}".
                format(trainAccuracy, testAccuracy, testStandardDeviation))
            self.calculateConfusionMatrixStats()
            print("TIME :: {}".format(end - start))
            print ('Tree end')
            print ()
