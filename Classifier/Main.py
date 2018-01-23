import pandas as pd


def readData():
    data = pd.read_csv("Data/ml1data.txt", sep=",", header=None)
    print(data.shape)

def main():
    readData()


if __name__ == '__main__':
    main()
