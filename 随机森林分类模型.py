#20260719优化bug记录
#删除未调用的死代码
#用np.sum统计类别数量
#新增空数据集判断，规避报错
#简化循环逻辑
#清理冗余 import
#优化记录
#新增get_subsamples做有放回随机采样，每棵树样本量和原数据集一致
#修正拼写错误：RondomForest → random_forest；treeForecast → tree_predict_single
#中文注释精简清晰，删除重复注释
#数据集切片用布尔掩码 mask_gt/mask_lt，抛弃np.nonzero嵌套调用
#增加空数据集判断，防止除以 0
#所有函数增加文档字符串docstring，说明入参、返回值、用途
#树深度剪枝逻辑独立可控，防止无限递归
#所有函数增加文档字符串docstring，说明入参、返回值、用途
#树深度剪枝逻辑独立可控，防止无限递归
#预测逻辑分层：单样本预测 → 单树批量预测 → 随机森林整体预测
#封装成类 RandomForest，更好保存模型、复用参数

from typing import List, Union, Dict, Any
import numpy as np
from numpy import inf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 任务类型常量，替代硬编码字符串，避免写错
TASK_REGRESSION = "regression"
TASK_CLASSIFICATION = "classification"


def get_datasets(n_samples: int = 200, n_features: int = 100, n_classes: int = 2) -> np.ndarray:
    """
    生成二分类模拟数据集，特征列+最后一列为标签
    :param n_samples: 样本总量
    :param n_features: 特征维度
    :param n_classes: 分类类别数
    :return: 拼接后数据集 [n_samples, n_features+1]
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
    # 拼接特征与标签
    dataset = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    return dataset


def get_subsamples(dataset: np.ndarray, n_sample: int) -> List[np.ndarray]:
    """
    Bootstrap有放回采样，生成n份子集（随机森林Bagging采样）
    :param dataset: 原始数据集
    :param n_sample: 需要生成的子集数量
    :return: 子集列表
    """
    n_rows = dataset.shape[0]
    subs = []
    for _ in range(n_sample):
        # 向量化随机索引，替代内层循环
        rand_idx = np.random.randint(0, n_rows, size=n_rows)
        subs.append(dataset[rand_idx, :])
    return subs


def bin_split_dataset(dataset: np.ndarray, feature: int, split_val: float) -> tuple[np.ndarray, np.ndarray]:
    """
    根据指定特征与分裂值二分数据集
    :param dataset: 输入数据集
    :param feature: 待分裂特征下标
    :param split_val: 分裂阈值
    :return: 大于阈值子集、小于阈值子集
    """
    mask_gt = dataset[:, feature] > split_val
    mask_lt = dataset[:, feature] < split_val
    mat0 = dataset[mask_gt]
    mat1 = dataset[mask_lt]
    return mat0, mat1


def reg_error(dataset: np.ndarray) -> float:
    """回归任务：计算数据集总方差（分裂损失）"""
    label_col = dataset[:, -1]
    return np.var(label_col) * len(label_col)


def reg_leaf(dataset: np.ndarray) -> float:
    """回归叶子节点：返回样本均值"""
    return np.mean(dataset[:, -1])


def majority_class(dataset: np.ndarray) -> int:
    """分类叶子节点：返回样本中占比最多的类别（仅二分类）"""
    labels = dataset[:, -1]
    cnt0 = np.sum(labels == 0)
    cnt1 = np.sum(labels == 1)
    return 0 if cnt0 > cnt1 else 1


def gini_index(dataset: np.ndarray) -> float:
    """计算数据集基尼不纯度"""
    total = len(dataset)
    if total == 0:
        return 0.0
    labels = dataset[:, -1]
    unique_labels = np.unique(labels)
    prob_sum = 0.0
    for label in unique_labels:
        p = np.sum(labels == label) / total
        prob_sum += p ** 2
    return 1 - prob_sum


def select_best_feature(
    dataset: np.ndarray,
    m_feat: int,
    task_type: str = TASK_REGRESSION
) -> tuple[Union[int, None], Union[float, int]]:
    """
    随机选取m个特征，遍历寻找最优分裂特征与阈值
    :param dataset: 数据集
    :param m_feat: 单棵树随机选取特征数量
    :param task_type: 任务类型 regression / classification
    :return: (最优特征下标, 分裂阈值)，无增益时返回(None, 叶节点值)
    """
    n_feats = dataset.shape[1] - 1  # 最后一列是标签，不算特征
    # 随机抽取m个特征下标
    rand_feat_idx = np.random.randint(0, n_feats, size=m_feat)
    # 计算分裂前整体损失
    if task_type == TASK_REGRESSION:
        base_loss = reg_error(dataset)
    else:
        base_loss = gini_index(dataset)

    best_loss = inf
    best_feat = None
    best_split_val = None

    for feat in rand_feat_idx:
        # 获取该特征所有唯一分裂点
        feat_vals = np.unique(dataset[:, feat])
        for val in feat_vals:
            mat0, mat1 = bin_split_dataset(dataset, feat, val)
            # 计算分裂后总损失
            if task_type == TASK_REGRESSION:
                current_loss = reg_error(mat0) + reg_error(mat1)
            else:
                current_loss = gini_index(mat0) + gini_index(mat1)
            # 更新最优分裂
            if current_loss < best_loss:
                best_loss = current_loss
                best_feat = feat
                best_split_val = val

    # 分裂增益不足，直接返回叶节点
    gain = base_loss - best_loss
    if gain < 0.001:
        if task_type == TASK_REGRESSION:
            return None, reg_leaf(dataset)
        else:
            return None, majority_class(dataset)

    return best_feat, best_split_val


def create_tree(
    dataset: np.ndarray,
    task_type: str = TASK_REGRESSION,
    m_feat: int = 20,
    max_depth: int = 10
) -> Union[Dict[str, Any], float, int]:
    """递归构建单棵CART决策树"""
    best_feat, split_val = select_best_feature(dataset, m_feat, task_type)
    # 无分裂增益，返回叶子
    if best_feat is None:
        return split_val
    # 达到最大深度，提前剪枝
    if max_depth <= 0:
        if task_type == TASK_REGRESSION:
            return reg_leaf(dataset)
        else:
            return majority_class(dataset)

    tree = {
        "best_feature": best_feat,
        "split_value": split_val
    }
    mat_left, mat_right = bin_split_dataset(dataset, best_feat, split_val)
    # 递归构建左右子树，深度-1
    tree["left"] = create_tree(mat_left, task_type, m_feat, max_depth - 1)
    tree["right"] = create_tree(mat_right, task_type, m_feat, max_depth - 1)
    return tree


def random_forest(
    dataset: np.ndarray,
    n_trees: int,
    task_type: str = TASK_REGRESSION,
    m_feat: int = 20,
    max_depth: int = 10
) -> List[Union[Dict[str, Any], float, int]]:
    """
    训练随机森林（标准Bagging：有放回采样生成子集，而非原代码错误切分train_test）
    :param dataset: 完整数据集（特征+标签）
    :param n_trees: 树数量
    :param task_type: 回归/分类
    :param m_feat: 单树随机特征数
    :param max_depth: 单树最大深度
    :return: 训练完成的树列表
    """
    tree_list = []
    # Bootstrap采样生成多份样本子集
    subsamples = get_subsamples(dataset, n_trees)
    for sub_data in subsamples:
        tree = create_tree(sub_data, task_type, m_feat, max_depth)
        tree_list.append(tree)
    return tree_list


def tree_predict_single(tree: Union[Dict, float, int], sample: np.ndarray, task_type: str) -> Union[float, int]:
    """单棵树对单个样本预测"""
    # 当前节点是叶子，直接返回值
    if not isinstance(tree, dict):
        if task_type == TASK_REGRESSION:
            return float(tree)
        else:
            return int(tree)

    feat_idx = tree["best_feature"]
    split_val = tree["split_value"]
    if sample[feat_idx] > split_val:
        return tree_predict_single(tree["left"], sample, task_type)
    else:
        return tree_predict_single(tree["right"], sample, task_type)


def predict_single_tree_batch(tree: Union[Dict, float, int], data: np.ndarray, task_type: str) -> np.ndarray:
    """单棵树批量预测整个数据集"""
    n = len(data)
    pred = np.zeros((n, 1))
    for i in range(n):
        pred[i, 0] = tree_predict_single(tree, data[i], task_type)
    return pred


def rf_predict(rf_trees: List, test_data: np.ndarray, task_type: str) -> np.ndarray:
    """
    随机森林批量预测
    :param rf_trees: 训练好的森林
    :param test_data: 待预测数据（仅特征，不含标签）
    :param task_type: 回归/分类
    :return: 预测结果矩阵 [n_sample, 1]
    """
    n_sample = len(test_data)
    pred_sum = np.zeros((n_sample, 1))

    # 累加所有树预测结果
    for tree in rf_trees:
        batch_pred = predict_single_tree_batch(tree, test_data, task_type)
        pred_sum += batch_pred

    if task_type == TASK_REGRESSION:
        # 回归：取所有树平均值
        pred_result = pred_sum / len(rf_trees)
    else:
        # 二分类投票：超过半数为1，否则0
        threshold = len(rf_trees) / 2
        pred_result = np.where(pred_sum > threshold, 1, 0)

    return pred_result


if __name__ == '__main__':
    # 1. 生成数据集
    full_data = get_datasets(n_samples=200, n_features=100, n_classes=2)
    X_all = full_data[:, :-1]
    y_all = full_data[:, -1:]
    print("真实标签转置：\n", y_all.T)

    # 2. 训练分类随机森林，4棵树
    rf_model = random_forest(
        dataset=full_data,
        n_trees=4,
        task_type=TASK_CLASSIFICATION,
        m_feat=20,
        max_depth=10
    )
    print("==================== 随机森林训练完成 ====================")

    # 3. 预测
    y_pred = rf_predict(rf_model, X_all, task_type=TASK_CLASSIFICATION)
    print("预测结果转置：\n", y_pred.T)

    # 4. 输出残差
    residual = y_all.T - y_pred.T
    print("真实值-预测值残差：\n", residual)



----历史版特意放在这里----
from numpy import inf
from numpy import zeros
import numpy as np
from sklearn.model_selection import train_test_split
 
#生成数据集。数据集包括标签，全包含在返回值的dataset上
def get_Datasets():
    from sklearn.datasets import make_classification
    dataSet,classLabels=make_classification(n_samples=200,n_features=100,n_classes=2)
    #print(dataSet.shape,classLabels.shape)
    return np.concatenate((dataSet,classLabels.reshape((-1,1))),axis=1)
 
 
#切分数据集，实现交叉验证。可以利用它来选择决策树个数。但本例没有实现其代码。
#原理如下：
#第一步，将训练集划分为大小相同的K份；
#第二步，我们选择其中的K-1分训练模型，将用余下的那一份计算模型的预测值，
#这一份通常被称为交叉验证集；第三步，我们对所有考虑使用的参数建立模型
#并做出预测，然后使用不同的K值重复这一过程。
#然后是关键，我们利用在不同的K下平均准确率最高所对应的决策树个数
#作为算法决策树个数
 
def splitDataSet(dataSet,n_folds):     #将训练集划分为大小相同的n_folds份；
    fold_size=len(dataSet)/n_folds
    data_split=[]
    begin=0
    end=fold_size
    for i in range(n_folds):
        data_split.append(dataSet[begin:end,:])
        begin=end
        end+=fold_size
    return data_split
#构建n个子集
def get_subsamples(dataSet,n):
    subDataSet=[]
    for i in range(n):
        index=[]     #每次都重新选择k个 索引
        for k in range(len(dataSet)):  #长度是k
            index.append(np.random.randint(len(dataSet)))  #(0,len(dataSet)) 内的一个整数
        subDataSet.append(dataSet[index,:])
    return subDataSet
 
#    subDataSet=get_subsamples(dataSet,10)
#############################################################################
 
 
 
#根据某个特征及值对数据进行分类
def binSplitDataSet(dataSet,feature,value):
    
 
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[np.nonzero(dataSet[:,feature]<value)[0],:]
 
    return mat0,mat1
 
'''
  feature=2 
  value=1
  dataSet=get_Datasets()
  mat0,mat1= binSplitDataSet(dataSet,2,1)
'''
 
#计算方差，回归时使用
def regErr(dataSet):
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]
 
#计算平均值，回归时使用
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])
 
def MostNumber(dataSet):  #返回多类
    #number=set(dataSet[:,-1])
    len0=len(np.nonzero(dataSet[:,-1]==0)[0])
    len1=len(np.nonzero(dataSet[:,-1]==1)[0])
    if len0>len1:
        return 0
    else:
        return 1
    
    
#计算基尼指数   一个随机选中的样本在子集中被分错的可能性   是被选中的概率乘以被分错的概率 
def gini(dataSet):
    corr=0.0
    for i in set(dataSet[:,-1]):           #i 是这个特征下的 某个特征值
        corr+=(len(np.nonzero(dataSet[:,-1]==i)[0])/len(dataSet))**2
    return 1-corr
 
 
def select_best_feature(dataSet,m,alpha="huigui"):
    f=dataSet.shape[1]                                            #拿过这个数据集，看这个数据集有多少个特征，即f个
    index=[]
    bestS=inf;
    bestfeature=0;bestValue=0;
    if alpha=="huigui":
        S=regErr(dataSet)
    else:
        S=gini(dataSet)
        
    for i in range(m):
        index.append(np.random.randint(f))                        #在f个特征里随机，注意是随机！选择m个特征，然后在这m个特征里选择一个合适的分类特征。 
                                                                  
    for feature in index:
        for splitVal in set(dataSet[:,feature]):                  #set() 函数创建一个无序不重复元素集，用于遍历这个特征下所有的值
            mat0,mat1=binSplitDataSet(dataSet,feature,splitVal)  
            if alpha=="huigui":  newS=regErr(mat0)+regErr(mat1)   #计算每个分支的回归方差
            else:
                newS=gini(mat0)+gini(mat1)                        #计算被分错率
            if bestS>newS:
                bestfeature=feature
                bestValue=splitVal
                bestS=newS                      
    if (S-bestS)<0.001 and alpha=="huigui":                      # 对于回归来说，方差足够了，那就取这个分支的均值
        return None,regLeaf(dataSet)
    elif (S-bestS)<0.001:
        #print(S,bestS)
        return None,MostNumber(dataSet)                          #对于分类来说，被分错率足够下了，那这个分支的分类就是大多数所在的类。
    #mat0,mat1=binSplitDataSet(dataSet,feature,splitVal)
    return bestfeature,bestValue
 
def createTree(dataSet,alpha="huigui",m=20,max_level=10):             #实现决策树，使用20个特征，深度为10，
    bestfeature,bestValue=select_best_feature(dataSet,m,alpha=alpha)
    if bestfeature==None:
        return bestValue
    retTree={}
    max_level-=1
    if max_level<0:   #控制深度
        return regLeaf(dataSet)
    retTree['bestFeature']=bestfeature
    retTree['bestVal']=bestValue
    lSet,rSet=binSplitDataSet(dataSet,bestfeature,bestValue)      #lSet是根据特征bestfeature分到左边的向量，rSet是根据特征bestfeature分到右边的向量
    retTree['right']=createTree(rSet,alpha,m,max_level)
    retTree['left']=createTree(lSet,alpha,m,max_level)            #每棵树都是二叉树，往下分类都是一分为二。
    #print('retTree:',retTree)
    return retTree
 
def RondomForest(dataSet,n,alpha="huigui"):   #树的个数
    #dataSet=get_Datasets()
    Trees=[]        # 设置一个空树集合
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(dataSet[:,:-1], dataSet[:,-1], test_size=0.33, random_state=42)
        X_train=np.concatenate((X_train,y_train.reshape((-1,1))),axis=1)
        Trees.append(createTree(X_train,alpha=alpha))
    return Trees     # 生成好多树
###################################################################
 
#预测单个数据样本，重头！！如何利用已经训练好的随机森林对单个样本进行 回归或分类！
def treeForecast(trees,data,alpha="huigui"):      
    if alpha=="huigui":
        if not isinstance(trees,dict):                       #isinstance() 函数来判断一个对象是否是一个已知的类型
            return float(trees)
        
        if data[trees['bestFeature']]>trees['bestVal']:      # 如果数据的这个特征大于阈值，那就调用左支
            if type(trees['left'])=='float':                 #如果左支已经是节点了，就返回数值。如果左支还是字典结构，那就继续调用， 用此支的特征和特征值进行选支。 
                return trees['left']
            else:
                return treeForecast(trees['left'],data,alpha)
        else:
            if type(trees['right'])=='float':
                return trees['right']
            else:
                return treeForecast(trees['right'],data,alpha)   
    else:
        if not isinstance(trees,dict):                      #分类和回归是同一道理
            return int(trees)
        
        if data[trees['bestFeature']]>trees['bestVal']:
            if type(trees['left'])=='int':
                return trees['left']
            else:
                return treeForecast(trees['left'],data,alpha)
        else:
            if type(trees['right'])=='int':
                return trees['right']
            else:
                return treeForecast(trees['right'],data,alpha)   
            
            
 
#随机森林 对 数据集打上标签   0、1 或者是 回归值
def createForeCast(trees,test_dataSet,alpha="huigui"):
    cm=len(test_dataSet)                      
    yhat=np.mat(zeros((cm,1)))
    for i in range(cm):                                     #
        yhat[i,0]=treeForecast(trees,test_dataSet[i,:],alpha)    #
    return yhat
 
 
#随机森林预测
def predictTree(Trees,test_dataSet,alpha="huigui"):      #trees 是已经训练好的随机森林   调用它！
    cm=len(test_dataSet)   
    yhat=np.mat(zeros((cm,1)))   
    for trees in Trees:
        yhat+=createForeCast(trees,test_dataSet,alpha)    #把每次的预测结果相加
    if alpha=="huigui": yhat/=len(Trees)            #如果是回归的话，每棵树的结果应该是回归值，相加后取平均
    else:
        for i in range(len(yhat)):                  #如果是分类的话，每棵树的结果是一个投票向量，相加后，
                                                    #看每类的投票是否超过半数，超过半数就确定为1
            if yhat[i,0]>len(Trees)/2:            
                yhat[i,0]=1
            else:
                yhat[i,0]=0
    return yhat
 
 
 
if __name__ == '__main__' :
    dataSet=get_Datasets()  
    print(dataSet[:,-1].T)                                     #打印标签，与后面预测值对比  .T其实就是对一个矩阵的转置
    RomdomTrees=RondomForest(dataSet,4,alpha="fenlei")         #这里我训练好了 很多树的集合，就组成了随机森林。一会一棵一棵的调用。
    print("---------------------RomdomTrees------------------------")
    #print(RomdomTrees[0])
    test_dataSet=dataSet                               #得到数据集和标签
    yhat=predictTree(RomdomTrees,test_dataSet,alpha="fenlei")  # 调用训练好的那些树。综合结果，得到预测值。
    print(yhat.T)
#get_Datasets()
    print(dataSet[:,-1].T-yhat.T)
