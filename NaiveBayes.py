import pandas as pd
import matplotlib as plt
from sys import argv


def loadData(train_data, train_labels, test_data, test_labels):
    train_data = pd.read_csv(train_data, delimiter=',', index_col=None, engine='python')
    train_labels = pd.read_csv(train_labels, delimiter=',', index_col=None, engine='python')
    df = pd.concat([train_data, train_labels], axis=1, ignore_index=False)
    test_data = pd.read_csv(test_data, delimiter=',', index_col=None, engine='python')
    test_labels = pd.read_csv(test_labels, delimiter=',', index_col=None, engine='python')
    for column in train_data.columns:
        train_data[column].fillna(train_data[column].mode()[0], inplace=True)
    return df, test_data, test_labels, train_labels, train_data


# Find the counts for those that survived and those that died for a particular attribute in the dataframe.
def findUnique(attr, df):
    dict = {}
    column = df[attr]
    vals = column.unique()
    sOd = df['survived']
    data = pd.concat([column, sOd], axis=1, ignore_index=False)
    for val in vals:
        dat = data[data[attr] == val]
        surCounts = 0
        deadCounts = 0
        for index, row in dat.iterrows():
            if row['survived'] == 0:
                deadCounts += 1
            else:
                surCounts += 1
        dict.update({val: (surCounts, deadCounts)})
    dict.update({'k': vals.size})
    return dict

# Finds the probability of having survived or died for each attribute's values and stores it in a dictionary
def findProbs(dict, df):
    d = {}
    surCounts = sum(df[df['survived'] == 1].count())
    deadCounts = sum(df[df['survived'] == 0].count())
    k = dict['k']
    for key in dict:
        if key == 'k':
            continue
        else:
            value = dict[key]
            surs = value[0] + 1
            dead = value[1] + 1
            denomS = surCounts + k
            denomD = deadCounts + k
            prob_they_survived = surs/denomS
            prob_they_died = dead/denomD
            d.update({key: (prob_they_survived, prob_they_died)})
    d.update({'default': (1/surCounts, 1/deadCounts)})
    #print(d)
    return d



# Multiplies all the elements of a list to each other
def multList(arr):
    fin = 1
    for i in arr:
        fin *= i
    return fin

# Predicts the value for each row in the test data
def predict(test_data, dict):
    fin = []
    for index, row in test_data.iterrows():
        cols = list(test_data.columns.values)
        surProbs = []
        deadProbs = []
        for i in range(0, len(row)):
            attr = cols[i]
            attribute = dict[attr]
            if attribute.get(row[i]) is not None:
                key1 = attribute[row[i]]
            else:
                key1 = attribute['default']
            surProbs.append(key1[0])
            deadProbs.append(key1[1])
        surProb = multList(surProbs)
        deadProb = multList(deadProbs)

        if surProb >= deadProb:
            fin.append(1)
        else:
            fin.append(0)
    return fin


# Tests the accuracy of the predictions
def testAccuracy(finarr, labels):
    correct = 0
    total = labels.size
    for i in range(labels.size):
        if labels.values[i] == finarr[i]:
            correct += 1
    return correct/total


# Calculate zero-one loss
def zeroOneLoss(test_labels, finarr):
    n = test_labels.size
    sum = 0
    for i in range(0, n):
        if test_labels.values[i] == finarr[i]:
            sum += 0
        else:
            sum += 1
    return sum/n

#Calculates the squared loss
def squaredLoss(dictionary):
    n = test_labels.size
    sum = 0
    for i in dictionary.values():
        if i[0] > i[1]:
            value = i[0]/(i[0] + i[1])
        else:
            value = i[1] / (i[0] + i[1])
        sum += value - 1
    return abs(sum/ n)


# Perceptron
def perceptron(train_data, train_labels, numOfIter):
    train_data = normalize(train_data)
    bias = 0.0
    weights = [0] * 7
    labels = train_labels['survived']
    for i in range(numOfIter):
        for index, row in train_data.iterrows():
            row = row.values
            error = labels[index] - predict2(row, weights, bias)
            if error == 0:
                continue
            else:
                bias = bias + error
                for c in range(0, len(train_data.columns)):
                    weights[c] = weights[c] + error * row[c]
    return weights, bias

def normalize(train_data):
    means = []
    std_devs = []

    for i in train_data.columns:
        column = train_data[i]
        means.append(column.std())

    for i in train_data.columns:
        column = train_data[i]
        std_devs.append(column.std())

    for index, row in train_data.iterrows():
        for i in range(0, len(train_data.columns)):
            row[i] = float(row[i] - means[i]) / float(std_devs[i])
        train_data.iloc[index] = row

    return train_data


# Predict for Perceptron
def predict2(row, weights, bias):
    pred = 0.0
    for i in range(0, len(row)):
        if not isinstance(row[i], str):
            pred += float(weights[i]) * float(row[i])
    pred += bias
    if pred >= 0:
        return 1
    else:
        return 0



__name__ = "__main__"

if __name__ == "__main__":

    train_data = argv[1]
    train_labels = argv[2]
    test_data = argv[3]
    test_labels = argv[4]


    df, test_data, test_labels, train_labels, train_data = loadData(train_data, train_labels, test_data, test_labels)
    #print(train_data)


    for percent in [1, 10, 50]:
        number = (percent / 100) * len(df)
        newdf = df.sample(n=percent)


    dic = {}
    probs = {}
    cols = list(df.columns.values)
    for attr in cols:
        if attr == 'survived':
            continue
        else:
            uniq = findUnique(attr, df)
            probs = findProbs(uniq, df)
            dic.update({attr: probs})

    finarr = predict(test_data, dic)
    print("ZERO-ONE LOSS=" + str(zeroOneLoss(test_labels, finarr)))
    print("SQUARED LOSS=" + str(squaredLoss(probs)) + " Test Accuracy=" + str(testAccuracy(finarr, test_labels)))
    print()

    finarr2 = []

    weights, bias = perceptron(train_data, train_labels, 2)
    for index, row in test_data.iterrows():
        finarr2.append(predict2(row, weights, bias))

    print("Hinge LOSS=" + str(zeroOneLoss(test_labels, finarr2)))
    print("Test Accuracy=" + str(str(testAccuracy(finarr2, test_labels))))

