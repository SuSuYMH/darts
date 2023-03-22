
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



## 4、关于项目结构

model_search.py是最基本的操作定义，MixOp、Cell、Network

model.py是根据genotypes.py里面作者搜索好的架构参数加载后变成作者搜索好的网络

train_search.py是搜索架构用的

train.py是搜索完成后，加载并固定架构参数alpha训练w用的

## 5、两种cell的区别
Norm-Cell： 输入与输出的FeatureMap尺寸保持一致
Reduce-Cell： 输出的FeatureMap尺寸减小一半
Norm-Cell和Reduce-Cell的结构相同，不过操作不同。

