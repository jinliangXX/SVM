import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def getData(path):
    # 从csv文件加载数据
    dataset = pd.read_csv(path)
    # 验证数据格式
    print('*' * 50)
    print('打印前5行数据')
    print(dataset.head(5))
    print('*' * 50)
    # 获取age与EstimatedSalary为训练样本
    X = dataset.iloc[:, [2, 3]].values
    # 获取Purchased为label
    y = dataset.iloc[:, 4].values
    # 划分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=0)
    # 对x进行特征量化
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    getData('../datasets/Social_Network_Ads.csv')
