
## 1、async
async is a reserved keyword in Python >= 3.7 so it is a SyntaxError to use it in this way. The word async must be changed to non_blocking for your code to work on current versions of Python.





## 2、具体怎么训练的是分为两步进行的：

第一步：用darts进行architect search，通过validation performance选出来cells

对应train_search.py

第二步：用选出来的cells构建网络，从头开始训练，报告他们在测试集上的表现

对应train.py



## 3、关于优化器
train.py 这一部分和train_search.py的区别就是没有 那部分了，直接把 拿过来用：
model和model_search的区别也就在于cell 部分是把学到的权重直接拿来建网络：

更新alpha的优化器在architect.py的22行
更新w的优化器在train_search的85行