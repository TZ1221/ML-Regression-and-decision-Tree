from decisionTree import decisionTree
import pandas as pd
from Regression import Regression


def load(file):
    dataSet = pd.read_csv(file, header=None)
    return dataSet.drop_duplicates()


def changeAttribute(dataSet, attribute):
    uniqueValues = dataSet[attribute].unique()
    dataSet[attribute].replace(uniqueValues, range(len(uniqueValues)), inplace=True)
    return dataSet


if __name__ == '__main__':
    

  
    print("Iris Dataset")

    irisDataSet = load('iris.csv')
    irisTree = decisionTree(irisDataSet, 10, [0.05, 0.10, 0.15, 0.20], True)
    irisTree.validate()

    print()
    print('>>>>>>>>>>>>>>>>>>>>>>>>')
    print("Spambase Dataset")

    spambaseDataSet = load('spambase.csv')
    spambaseTree = decisionTree(spambaseDataSet, 10, [0.05, 0.10, 0.15, 0.20, 0.25],True)
    spambaseTree.validate()


    print()
    print('>>>>>>>>>>>>>>>>>>>>>>>>')
    print("Mushroom Dataset ---- Multiway Split")

    mushroomDataSet = load('mushroom.csv')
    mushroomDataSet = changeAttribute(mushroomDataSet,len(mushroomDataSet.columns) - 1)
    mushroomMultiwayTree = decisionTree(mushroomDataSet, 10, [0.05, 0.10, 0.15], False)
    mushroomMultiwayTree.validate()
    print()
    print('>>>>>>>>>>>>>>>>>>>>>>>>')

    print("Mushroom Dataset ---- Binary Split")
    mushroomModifiedDataSet = pd.get_dummies(data=mushroomDataSet, columns=range(len(mushroomDataSet.columns) - 1))
    targetAttributeColumn = mushroomModifiedDataSet[len(mushroomDataSet.columns) - 1]
    mushroomModifiedDataSet.drop(
        labels=[len(mushroomDataSet.columns) - 1], axis=1, inplace=True)
    mushroomModifiedDataSet.insert(
        len(mushroomModifiedDataSet.columns), len(mushroomDataSet.columns) - 1,
        targetAttributeColumn)
    mushroomBinaryTree = decisionTree(mushroomModifiedDataSet, 10, [0.05, 0.10, 0.15],True)
    mushroomBinaryTree.validate()

    print()
    print('>>>>>>>>>>>>>>>>>>>>>>>>')
    print("Housing Dataset")

    housingDataSet = load('housing.csv')
    housingRegression = Regression(housingDataSet, 10,[0.05, 0.10, 0.15, 0.20])
    housingRegression.validate()
