import fire
from sklearn.svm import SVC
from data import getData
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


def train(**kwargs):
    '''
    训练
    '''
    # 获取预处理后的数据集
    csv_path = 'datasets/Social_Network_Ads.csv'
    X_train, X_test, y_train, y_test = getData(csv_path)
    # 调用sklearn中的支持向量机
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)
    # 使用训练好的支持向量机预测测试集
    y_pred = classifier.predict(X_test)
    # 根据测试集的预测值和真实值构建混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print('*' * 50)
    print('混淆矩阵为：')
    print(cm)
    print('*' * 50)
    # 可视化
    visualization(classifier, X_train, y_train)


def visualization(model, X_train, y_train):
    '''
    可视化训练结果
    '''
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
        X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('SVM (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fire.Fire()
