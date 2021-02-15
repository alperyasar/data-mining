#1.kutuphaneler
import pandas as pd
from NaiveBayesClassifier import NaiveBayesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
import copy


def main():
    data = read_file("veriler.csv")
    data[1].pop(0)
    k = 6
    score = 0
    for i in range(0,k):
        x_train,y_train,x_test,y_test = k_validation(data[0],data[1],i,k)
        NB = NaiveBayesClassifier()
        NB.train(x_train, y_train)
        predicted_classes = NB.predict(x_test)

        cm = confusion_matrix(y_test, predicted_classes)
        score += take_Fscore(cm)
    score /= k
    print("\n\n","*"*60,"\n")

    print('f1 score: '+ '{0:.2f}'.format(score * 100))
    print("\n","*"*60)
    
    PCArun()
    LDArun()

    
def read_file(csv_path):
    """Reads and processes a csv data file. Returns a tuple of:
    (<2D list of instances>, <list of class labels>, <number of unique labels>).
    """
    
    df = pd.read_csv(csv_path, header=None)
    # Add a list of each instance for each attribute (the first N-1 columns in the DataFrame)
    instance_list = []
    if ((len(df.columns) > 1)):
        for attribute_index in range(0, (len(df.columns) - 1)):
            instance_list.append(df[attribute_index].tolist())
    for i in range(0, len(instance_list)):
        instance_list[i].pop(0)
    # Make sure attribute instances are in String format
    for index in range (0, len(instance_list)):
        instance_list[index] = [str(i) for i in instance_list[index]]
        
    class_list = []
    if ((len(df.columns) > 0)):
        class_list = df[(len(df.columns) - 1)].tolist()
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


def PCArun():
    ######################################################################
    veriler = pd.read_csv('veriler.csv')
    x_train = veriler.iloc[:,1:4].values #bağımsız değişkenler
    y_train = veriler.iloc[:,4:].values #bağımlı değişken
    pca = PCA()
    pca = PCA(n_components=1)
    sc=StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_train = pca.fit_transform(X_train)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    scores = cross_val_score(classifier, X_train, y_train,cv=7)
    print("\n\n","*"*60,"\n")

    print('PCA f1 score: '+ '{0:.2f}'.format(scores.mean() * 100))
    print("\n","*"*60)

    
def LDArun():
    ######################################################################
    veriler = pd.read_csv('veriler.csv')
    x_train = veriler.iloc[:,1:4].values #bağımsız değişkenler
    y_train = veriler.iloc[:,4:].values #bağımlı değişken
    lda = LDA()
    lda = LDA(n_components=1)
    sc=StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_train = lda.fit_transform(X_train,y_train)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    scores = cross_val_score(classifier, X_train, y_train,cv=7)
    print("\n\n","*"*60,"\n")
    print('LDA f1 score: '+ '{0:.2f}'.format(scores.mean() * 100))
    print("\n","*"*60)
    
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
    

main()
