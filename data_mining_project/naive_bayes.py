import pandas as pd
from NaiveBayesClassifier import NaiveBayesClassifier
from sklearn.metrics import confusion_matrix
import copy

def naiveBayes():
    print("old Naive Bayes algorithm accurance average : ",old_naiveBayes())


    score_predict()
#    k = score_predict()
#    print("improved Naive Bayes algorithm accurance average : ",k[0])
    

def old_naiveBayes():
    i = 0
    averageAcc = 0
    for k in range(4,7):
        if k == 5:
            continue
        i += 1
        data = read_file("lig.csv",k)
        data1 = read_file("tahmin.csv",k)
        data[1].pop(0)
        data1[1].pop(0)
        if k == 4:
            eksikVeri1(data[1])
        elif k == 6:
            eksikVeri(data[1])
        x_train,y_train = data[0],data[1]
        x_test,y_test = data1[0],data1[1]
        NB = NaiveBayesClassifier()
        NB.train(x_train, y_train)
        predicted_classes = NB.predict(x_test)
        print("*"*100,"\n")
        print("Tested class :   ",y_test)
        print("Predicted Class :",predicted_classes,"\n")
        print("Confussion Matrix :")
        print(confusion_matrix(y_test, predicted_classes),"\n")
        getA = getAccuracy(y_test, predicted_classes)
        averageAcc += getA
        print("Accuracy of naive Bayes : %",getA,"\n")
        print("*"*100)
    return averageAcc / i

def score_predict():
    j = []
    for i in range(0,2):

        print("\n")
        print("#"*100)
        print("#"*100,"\n\n")
        data = read_file("lig.csv",5)
        data1 = read_file("tahmin.csv",5)
        data[1].pop(0)
        data1[1].pop(0)
        x_train,y_train = data[0],data[1]
        x_test = data1[0]

        if i == 1:
            predicted_classes = []
            for j in range(0,len(data1[1])):
                temp = [[x_test[0][j]],[x_test[1][j]],[x_test[2][j]],[x_test[3][j]]]
                teams,y = reduceData(x_train, temp,y_train)
                NB = NaiveBayesClassifier()
                NB.train(teams, y)
                temp2 = NB.predict(temp)
                predicted_classes.append(temp2[0]) 
            print("Improved and reduced data Naive Bayes algorithm accurance average : ",improveScorePredict1(predicted_classes))
                
        else:
            NB = NaiveBayesClassifier()
            NB.train(x_train, y_train)
            predicted_classes = NB.predict(x_test)

            print("Improved Naive Bayes algorithm accurance average : ",improveScorePredict1(predicted_classes))
        print("*"*100)
#        j.append(improveScorePredict1(predicted_classes))
#    return j
    
def improveScorePredict1(predScore):
    winner = []
    goal = []
    for i in range(0,len(predScore)):
        x,y = predScore[i].split("-")
        x = int(x)
        y = int(y)
        if x > y:
            winner.append('Home')
        elif x == y:
            winner.append('Draw')
        else : winner.append('Away')
        if x+y >= 3:
            goal.append('Over')
        else : goal.append('Under')
    
    data1 = read_file("tahmin.csv",4)
    data1[1].pop(0)    
    y_test = data1[1]
    averageAcc = 0
    print("*"*100,"\n")
    print("Tested class :   ",y_test)
    print("Predicted Class :",winner,"\n")
    print("Confussion Matrix :")
    print(confusion_matrix(y_test, winner),"\n")
    getA = getAccuracy(y_test, winner)
    averageAcc += getA
    print("Accuracy of naive Bayes : %",getA,"\n")
    print("*"*100)
    
    data1 = read_file("tahmin.csv",6)
    data1[1].pop(0)    
    y_test = data1[1]
    print("*"*100,"\n")
    print("Tested class :   ",y_test)
    print("Predicted Class :",goal,"\n")
    print("Confussion Matrix :")
    print(confusion_matrix(y_test, goal),"\n")
    getA = getAccuracy(y_test, goal)
    averageAcc += getA
    print("Accuracy of naive Bayes : %",getA,"\n")
    print("*"*100)
    
    return averageAcc / 2

def reduceData(set1,set2,y):
    y_test = []
    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    for k in range(0,20):
        for i in range(0,len(y)):
            if set1[0][i] == set2[0][0] or set1[0][i] == set2[1][0] or set1[1][i] == set2[0][0] or set1[1][i] == set2[1][0]:
                temp1.append(set1[0][i])
                temp2.append(set1[1][i])
                temp3.append(set1[2][i])
                temp4.append(set1[3][i])
                y_test.append(y[i])
    teams = [temp1,temp2,temp3,temp4]
    return teams,y_test
def read_file(csv_path,k):
    """Reads and processes a csv data file. Returns a tuple of:
    (<2D list of instances>, <list of class labels>, <number of unique labels>).
    """
    
    df = pd.read_csv(csv_path, header=None)
    # Add a list of each instance for each attribute (the first N-1 columns in the DataFrame)
    instance_list = []
    if ((len(df.columns) > 1)):
        for attribute_index in range(0, 4):
            instance_list.append(df[attribute_index].tolist())
    for i in range(0, len(instance_list)):
        instance_list[i].pop(0)
    # Make sure attribute instances are in String format
    for index in range (0, len(instance_list)):
        instance_list[index] = [str(i) for i in instance_list[index]]
        
    class_list = []
    if ((len(df.columns) > 0)):
        class_list = df[k].tolist()
    class_list = [str(i) for i in class_list]
    
    n_classes = len(set(class_list))
    return instance_list, class_list, n_classes
def take_Fscore(cm):
    """Evaluates the number of correct predictions made by a Multinomial Naive Bayes classifier.
    Returns an accuracy score between [0,1].
    """
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    presicion = TP / (TP+FP)
    recall = TP / (TP+FN)

    return 2*presicion*recall / (presicion + recall)

    
def k_validation(x,y,n,k):
    x_train = copy.deepcopy(x)
    y_train = copy.deepcopy(y)
    n = int(len(y)/k) * n
    if n + int(len(y)/k) > len(y):
        m = len(y)
    else: 
        m = n + int(len(y)/k)
    
    x_test = []
    y_test = []
    for i in range(n,m):
        y_test.append(y_train[n])
        y_train.pop(n)
    for i in range(0,len(x)):
        z = []
        for j in range(n,m):
            z.append(x_train[i][n])
            x_train[i].pop(n)
        x_test.append(z)
    return x_train,y_train,x_test,y_test

#Get Accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet)))*100.0



def eksikVeri(y_test):
    veriler = pd.read_csv('lig.csv')
    y_train = veriler.iloc[:,5:6].values #bağımlı değişken

    for i in range(0,len(y_train)):
        x,y = y_train[i][0].split('-')
        x = int(x)
        y = int(y)
        if x + y >= 3 :
            if (y_test[i] != "Over"):
                y_test[i] = "Over"
        else : 
            if (y_test[i] != "Under"):
                y_test[i] = "Under"


def eksikVeri1(y_test):
    veriler = pd.read_csv('lig.csv')
    y_train = veriler.iloc[:,5:6].values #bağımlı değişken
    for i in range(0,len(y_train)):
        x,y = y_train[i][0].split('-')
        x = int(x)
        y = int(y)
        if x > y:
            if (y_test[i] != "Home"):
                y_test[i] = "Home"
        elif x < y:
            if (y_test[i] != "Away"):
                y_test[i] = "Away"
        else : 
            if (y_test[i] != "Draw"):
                y_test[i] = "Draw"
