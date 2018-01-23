import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

def read_and_process_data():
    data = pd.read_csv("Data/ml1data.txt", sep=",", index_col=0,names=["x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x18","x19","x20","y"])
    print(data.shape)
    data_dummy = pd.get_dummies(data,drop_first=True) # Perform One-Hot Encoding with drop first to avoid trap
    print(data_dummy.shape,'\n' )
    X = data_dummy.iloc[:,0:data_dummy.shape[1]-1]
    y = data_dummy['y_MEM'].values


def main():
    read_and_process_data()


if __name__ == '__main__':
    main()
