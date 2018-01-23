import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.naive_bayes import GaussianNB

def read_and_process_data():
    data = pd.read_csv("Data/ml1data.txt", sep=",", index_col=0, names=["x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x18","x19","x20","y"])
    print(data.shape)
    data_dummy = pd.get_dummies(data) # Perform One-Hot Encoding with drop first to avoid trap
    print(data_dummy.columns,'\n' )
    columns = []
    for x in range(0,data_dummy.columns.size-2):
        str = data_dummy.columns[x]
        columns.append(str)
    X = data_dummy[columns].values
    y = data_dummy['y_MEM'].values
    return dict(input=X, output=y)


def perform_logicticregression(X,y):
    logreg = LogisticRegression()
    scores = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
    print(scores.mean())


def perform_linerarregression(X,y):
    linreg = LinearRegression()
    scores = cross_val_score(linreg, X, y, cv=10, scoring='accuracy')
    print(scores.mean())


def perform_naivebias(X,y):
    nb = GaussianNB()
    scores = cross_val_score(nb, X, y, cv = 10, scoring= 'accuracy')

    print(scores.mean())


def main():
    data = read_and_process_data()
    perform_logicticregression(data["input"],data["output"])
   # perform_linerarregression(data["input"],data["output"])
    perform_naivebias(data["input"], data["output"])


if __name__ == '__main__':
    main()
