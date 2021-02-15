#1.kutuphaneler
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def svm():
    
    #2.veri onisleme
    #2.1.veri yukleme
    veriler = pd.read_csv('lig.csv')
    veriler1 = pd.read_csv('tahmin.csv')
    #pd.read_csv("veriler.csv")
    #test
    x_train = veriler.iloc[:,0:4].values #bağımsız değişkenler
    y_train = veriler.iloc[:,4:5].values #bağımlı değişken
    y_train1 = veriler.iloc[:,6:7].values #bağımlı değişken
    eksikVeri(y_train,y_train1)
 
    x_test = veriler1.iloc[:,0:4].values #bağımsız değişkenler
    y_test = veriler1.iloc[:,4:5].values #bağımlı değişken
    y_test1 = veriler1.iloc[:,6:7].values #bağımlı değişken
    #verilerin egitim ve test icin bolunmesi
    convert_teams(x_train, x_test)
    convert_date(x_train, x_test)
    x = 0.0
    pred = old(x_test, x_train, y_train)
    x += printScreen(y_test, pred)
    pred = old(x_test, x_train, y_train1)
    x = (printScreen(y_test1, pred) + x) / 2
    
    print("old SVM algorithm accurance average : ",x)
    print("*"*100,"\n")
    print("#"*100)
    print("#"*100,"\n\n")

    x = 0.0
    pred = improved(x_test, x_train, y_train)
    x += printScreen(y_test, pred)
     
    pred = improved(x_test, x_train, y_train1)
    x = (printScreen(y_test1, pred) + x) /2
    print("Improved SVM algorithm accurance average : ",x)
    print("*"*100,"\n")
    improvedByScored()

def improvedByScored():
        #2.veri onisleme
    #2.1.veri yukleme
    veriler = pd.read_csv('lig.csv')
    veriler1 = pd.read_csv('tahmin.csv')
    #pd.read_csv("veriler.csv")
    #test
        #test
    x_train = veriler.iloc[:,0:4].values #bağımsız değişkenler
    y_train = veriler.iloc[:,5:6].values #bağımlı değişken

    x_test = veriler1.iloc[:,0:4].values #bağımsız değişkenler
    y_test = veriler1.iloc[:,4:5].values #bağımlı değişken
    y_test1 = veriler1.iloc[:,6:7].values #bağımlı değişken
    #verilerin egitim ve test icin bolunmesi
    convert_teams(x_train, x_test)
    convert_date(x_train, x_test)
    pred = improved(x_test, x_train, y_train)
    pred_y1 = []
    pred_y2 = []
    for i in pred:
        x,y = i[0].split('-')
        x = int(x)
        y = int(y)
        if x > y:
            pred_y1.append(['Home'])
        elif x < y:
            pred_y1.append(['Away'])
        else : pred_y1.append(['Draw'])
        if x + y >= 3 :
            pred_y2.append(['Over'])
        else : pred_y2.append(['Under'])
        
    x = 0.0
    x += printScreen(y_test, pred_y1)

    x = (printScreen(y_test1, pred_y2) + x) /2
    print("Improved SVM by Scored algorithm accurance average : ",x)
    print("*"*100,"\n")

def old(x_test,x_train,y_train):
    sc=StandardScaler()
    
    X_train = sc.fit_transform(x_train)
    X_test = sc.transform(x_test)
    
    logr = SVC(kernel = 'rbf', random_state = 0)
    logr.fit(X_train,y_train)
    
    return logr.predict(X_test)

def improved(x_test,x_train,y_train):
    pred = []
    for i in range(0,len(x_test)):
        
        teams,y = reduceData(x_train, x_test[i],y_train)
          
        sc=StandardScaler()
        
        X_train = sc.fit_transform(teams)
        X_test = sc.transform(x_test)
               
        logr = SVC(kernel = 'rbf', random_state = 0)
        logr.fit(X_train,y)
        x = [X_test[i]]
    
        pred.append(logr.predict(x))
    return pred

def printScreen(y_test,pred):
    print(pred)
    print(y_test)
    print("Confussion Matrix :")
    print(confusion_matrix(y_test, pred),"\n")
    result = getAccuracy(y_test, pred)
    print("Accuracy of SVM : %",result,"\n")
    print("*"*100)    
    return result
def reduceData(set1,set2,y):
    teams = []
    y_test = []
    for k in range(0,12):
        for i in range(0,len(set1)):
            if set1[i][0] == set2[0] or set1[i][0] == set2[1] or set1[i][1] == set2[0] or set1[i][1] == set2[1]:
                teams.append(set1[i])
                y_test.append(y[i][0])

    return teams,y_test
def convert_teams(set1,set2):
    teams = []
    for i in range(0,len(set1)):
        if set1[i][0] not in teams:
            teams.append(set1[i][0])
            set1[i][0] = len(teams)
        else: set1[i][0] = teams.index(set1[i][0]) + 1
        if set1[i][1] not in teams:
            teams.append(set1[i][1])
            set1[i][1] = len(teams)
        else: set1[i][1] = teams.index(set1[i][1]) + 1
    
    for i in range(0,len(set2)):
        if set2[i][0] not in teams:
            teams.append(set2[i][0])
            set2[i][0] = len(teams)
        else: set2[i][0] = teams.index(set2[i][0]) + 1
        if set2[i][1] not in teams:
            teams.append(set2[i][1])
            set2[i][1] = len(teams)
        else: set2[i][1] = teams.index(set2[i][1]) + 1
    return set1,set2
        
def convert_date(set1,set2):
    teams = []
    for i in range(0,len(set1)):
        if set1[i][2] not in teams:
            teams.append(set1[i][2])
            set1[i][2] = len(teams)
        else: set1[i][2] = teams.index(set1[i][2]) + 1
        if set1[i][3] not in teams:
            teams.append(set1[i][3])
            set1[i][3] = len(teams)
        else: set1[i][3] = teams.index(set1[i][3]) + 1
    
    for i in range(0,len(set2)):
        if set2[i][2] not in teams:
            teams.append(set2[i][2])
            set2[i][2] = len(teams)
        else: set2[i][2] = teams.index(set2[i][2]) + 1
        if set2[i][3] not in teams:
            teams.append(set2[i][3])
            set2[i][3] = len(teams)
        else: set2[i][3] = teams.index(set2[i][3]) + 1

#Get Accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet)))*100.0




def eksikVeri(y_test,y_test1):
    veriler = pd.read_csv('lig.csv')
    y_train = veriler.iloc[:,5:6].values #bağımlı değişken
    for i in range(0,len(y_train)):
        x,y = y_train[i][0].split('-')
        x = int(x)
        y = int(y)
        if x > y:
            if (y_test[i][0] != "Home"):
                y_test[i][0] = "Home"
        elif x < y:
            if (y_test[i][0] != "Away"):
                y_test[i][0] = "Away"
        else : 
            if (y_test[i][0] != "Draw"):
                y_test[i][0] = "Draw"
        if x + y >= 3 :
            if (y_test1[i][0] != "Over"):
                y_test1[i][0] = "Over"
        else : 
            if (y_test1[i][0] != "Under"):
                y_test1[i][0] = "Under"




