# **SVM**

通过sklearn实现支持向量机，并通过数据集进行效果验证。

### 目录结构

- data：数据集预处理部分
- datasets：包含scv格式的数据集
- main.py：程序执行的入口

### 安装相关包

```shell
pip install -r requirements.txt
```

### 程序运行方式

```
python main.py train
```

### 运行结果

```
**************************************************
打印前5行数据
    User ID  Gender  Age  EstimatedSalary  Purchased
0  15624510    Male   19            19000          0
1  15810944    Male   35            20000          0
2  15668575  Female   26            43000          0
3  15603246  Female   27            57000          0
4  15804002    Male   19            76000          0
**************************************************
/home/xujinliang/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py:475: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
  warnings.warn(msg, DataConversionWarning)
**************************************************
混淆矩阵为：
[[63  5]
 [ 7 25]]
**************************************************
```

![image-20190523200250322](https://jinliangxx.oss-cn-beijing.aliyuncs.com/2019-05-23-120250.png)