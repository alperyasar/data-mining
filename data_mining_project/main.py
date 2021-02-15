#1.kutuphaneler
from naive_bayes import naiveBayes
from lr import lr
from SVM import svm


def main():
    print("\n\n\n")
    print("Naive Bayes running...\n\n\n")
    naiveBayes()
    print("\n\n\n")
    print("Logistic Regression running...\n\n\n")
    lr()
    print("\n\n\n")
    print("Support Vector Machine running...\n\n\n")
    svm()
main()