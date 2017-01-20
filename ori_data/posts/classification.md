---
title: Image Classification -- Data-driven Approach, k-Nearest Neighbor, train/val/test splits
date: 2016-01-16 22:04:43
tags: cs231n
---

## 前言
本文翻译的是这篇[CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/classification/)文章。

## 图像分类
**原因** 这一节我们将介绍图像分类问题，这是一个从一组固定类别中指定输入图像标签的任务。这是计算机视觉中的一个核心问题，尽管它简单，在实际中有各种各样的应用。此外，我们会在使用过程中看到，许多其他看似不同的计算机视觉任务（如目标检测，分割）可以归结为图像分类。

**例子** 例如，下图中一个图像分类模型获取一张图片然后指定四个标签的*{cat, dog, hat, mug}*概率。如图所示，请记住，图像在计算机中表示为一个大的三维数组。在这个例子中，猫的图片宽度为248像素，高度为400像素，并且有三个颜色通道：Red，Green，Blue（或者缩写为RGB）。因此，图像由248×400×3或总共297600个数字组成。每个数字是一个整数，范围从0（黑色）到255（白色）。我们的任务是将百万数据中得四分之一标记为单一的标签，如“猫”。

- - -
![classify](http://cs231n.github.io/assets/classify.png)
<font size=2>*图像分类*的任务是对于一个给定的图像预测一个标签（或这里显示的不同标签下概率分布，用以表示我们的信任）。图像是整数三维数组，整数的范围从0到255，图像的大小是宽度x高度x3。3表示三个颜色通道：Red，Green，Blue。</font>
- - -
<!-- more -->
**挑战** 由于识别视觉概念（如猫）这个任务对于人来说非常容易做到，这是一个值得思考的涉及到计算机视觉算法视角的挑战。正如我们以下提出的一系列挑战，请记住，图像的原始表示是一个三维阵列:

* *视点变化*  一个物体的可以多个角度观察。
* *比例变化*  可见的物体通常在尺寸上表现出变化（现实世界中的尺寸，不仅是其在图像中的区域）。
* *形变*  许多关注的对象不是刚体，可以以极端的方式变形。
* *遮挡*  关注的对象会被遮挡。有时可能一个对象中的一小部分（甚至几个像素）是可见的。
* *光照条件*  光照的影响在像素级是非常剧烈的。
* *背景混乱*  关注的对象可能融入周围的环境，使他们难以确定。
* *类内变化*  关注的对象往往是比较宽泛的，比如椅子。这些对象中有许多不同的类型，每种类型都有自己的外观。

一个好的图像分类模型所有变量的叉积必须是不变的，同时对于类内变化保持敏感性。

- - -
![challenges](http://cs231n.github.io/assets/challenges.jpeg)
- - -

**数据驱动方法** 我们要写一个什么样的算法才能将图像分为不同的类别？例如，不像写一个排序算法，人们写一个用于识别图像中猫的算法并不是显而易见的。因此，我们采取的方法和你会对小孩子采用的方法一样，而不是试图直接在代码中指定关注的物体的每一类是什么样的。我们给计算机提供每一类物体的很多样本，然后制定学习算法，看看这些例子，了解每个物体的视觉外观。这种方法被称为*数据驱动方法*，它依赖于已标记图像的*训练数据集*的最初积累。这里有一个关于这种数据集可能的例子：
- - -
![trainset](http://cs231n.github.io/assets/trainset.jpg)
<font size=2>这是四个类别训练集的例子。在实践中，我们可能有数以千计的类别，每个类别有成千上万个图像。</font>
- - -

**图像分类流程** 我们看到，图像分类的任务是，输入代表一张图片的像素数组，并为其分配一个标签。我们完整的流程可以按以下形式表示：

* **输入** 我们的输入*N*个图像，每个图像标为*K*个不同的类别中的一个。我们将此数据作为*训练集*。
* **学习** 我们的任务就是利用训练集学习每一个类别是什么样子。我们把这个步骤叫做*训练分类器*，或*学习一个模型*。
* **评估** 最后，我们通过让分类器预测一组它从未见过的新图像来评估分类器的质量。我们将用这些图像的真实标签和那些由分类预测的标签进行比较。直观地说，我们希望有大量的预测同真正答案（我们称之为基础事实）相匹配。

## 近邻分类器
作为我们的第一个方法，我们将开发一个被称为**近邻分类器**的算法。这种分类和卷积神经网络没有关系，并且在实践中很少使用，但它使我们明白图像分类问题的基本方法。

**图像分类数据集：CIFAR-10示例** 一种流行的微型图像分类数据集[CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)。此数据集包括60,000个微小的图像，这些图像是32像素x32像素。每个图像都标为十类（例如“飞机，汽车，鸟等”）中的一类。这60,000个图像被划分成包含50,000个图像的训练集和包含10,000个图像的测试集。下图中可以看到从十个类的每个类中都随机选取十个示例图片：

- - -
![nn](http://cs231n.github.io/assets/nn.jpg)
<font size=2>左：[CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)中的示例图片，右：第一列显示了一些测试图像，接下来，根据根据逐像素差我们显示这些测试图像在训练集中前10个近邻。</font>
- - -
假设现在我们得到CIFAR-10训练集中的50,000个图像（每个标签有5,000个图像），我们希望标注剩余的10,000。最近邻分类器将测试图像同训练集中的的每一个图像比较，并预测与其最接近的训练集图像的标签。在上图右侧，你可以看到10个测试图像处理后的结果。注意，仅约3/10的图像被正确检索到，而剩余的7/10没有被正确检索。例如，第八行中与马的头部最近邻的训练图像是一部红色的汽车，大概是由于汽车背景过于黑。其结果导致了一匹马在这种情况下被被误认为成一辆车。

你可能已经注意到，我们并未详细说明是如何比较两幅图片的，这个例子中图片的大小都是32×32×3。其中一个最简单的办法是按照像素进行比较，并把差值相加。换句话说，给定的两个图片并把他们当作矢量$I_1$，$I_2$，比较这两个图片一个合理的选择是**L1 distance**：
$$
d_1 (I_1, I_2) = \sum_p \left| I^p_1 - I^p_2 \right|
$$
这里把所有差值相加，一下是具体的步骤：
- - -
![nneg](http://cs231n.github.io/assets/nneg.jpeg)
<font size=2>这是采用逐像素差异来比较两个图像的*L1 distance*（这是一个颜色通道）的一个例子。两个图像按像素相减，然后所有差异相加。如果两个图像相同，结果将是零。但是如果图像有很大的不同，结果将非常大。</font>
- - - 
让我们看看如何用代码实现分类器。首先，加载CIFAR-10数据集到内存中，存储为4个数组：训练数据集及标签和测试数据集及标签。在下面的代码，`Xtr`（大小是50,000×32×32×3）存储训练集中的所有图像，相应的1维阵列`Ytr`（长度50,000）存储训练集的标签（从0到9）：

```python
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
```
现在，我们把所有的图像都延展成行向量，以下是我们如何训练、评估分类器：

```python
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```
注意，作为评价的标准，通常使用*精确度*衡量正确预测的百分比。请注意，我们要建立的所有分类器，将使用这一个通用的API：`train(X,y)`，该函数获取数据及相应标签用以训练。分类器应该建立某种标签的模型，该模型可以用数据进行预测。接着一个`predict(X)`函数，该函数获取新的数据并预测其标签。当然，我们已经忽略了事物的主体--分类器本身。下面是一个简单的*L1 distance*近邻分类器的实现模板：

```python
import numpy as np

class NearestNeighbor:
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```

如果你运行这段代码，你会看到该分类器在CIFAR-10上只有38.6％的精确度。这是比随机猜测（10％的准确度，因为有10个类）要好一点，但是远不及人的行为（估计[约为94％](http://karpathy.github.io/2011/04/27/manually-classifying-cifar10/)）或最好的卷积神经网络可以达到95％，和人的准确率相匹配（参考Kaggle比赛上最近CIFAR-10的[排行榜](https://www.kaggle.com/c/cifar-10/leaderboard)）。
**选择distance** 有许多计算两个向量之间距离的方法。另一种常见的选择是*L2 distance*，其几何解释就是计算两个向量之间的欧几里得距离。公式如下：
$$
d_2 (I_1, I_2) = \sqrt{\sum_p \left( I^p_1 - I^p_2 \right)^2}
$$
换句话说，我们会像以前那样计算逐像素的差异，但这次我们将所有差异值取平方然后相加，最后取平方根。在numpy中，使用上面代码我们只需要替换其中的一行。下面这行计算*L2 distance*：
```python
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```
注意，上述代码调用了`np.sqrt`，但在一个实用的近邻应用中，我们可以忽略求平方根操作，因为平方根是一个单调函数。也就是说，它扩展了距离的绝对值大小，但它保留了排序，所以有没有取平方根近邻都是相同的。如果你用这个距离的近邻分类器在CIFAR-10上运行，你将获得35.4％的准确度（略低于L1 distance的结果）。
**L1 vs. L2** 考虑两个度量之间的差异很有意义。具体地，当涉及到两个向量之间的差异时L2比L1更严厉。也就是说L2相对于中等的差异更喜欢大的差异。L1和L2是p-norm特殊情况中最常用的。

### k-近邻分类

你可能已经注意到了，当我们做预测时，只用最近邻图像的标签，这非常奇怪。事实上，使用K-近邻分类器几乎总是可以得到更好的结果。我们的想法很简单：找到训练集中前k个最接近的图像，而不是找训练集中最接近的图像，并让它们和测试图像的标签一一对应。特别是，当k= 1时，得到最近邻分类器。直观地看，K值越高产生平滑作用，使分类更加耐异常值：

- - -
![knn](http://cs231n.github.io/assets/knn.jpeg)
<font size=2>这是使用2维点和3个类（红，蓝，绿）展示的最近邻和5-近邻分类器之间的差异的例子。着色区域显示的决策边界是由使用L2 distance分类器产生的。白色区域中的点分类不清晰（即图中的点至少和两个类对应）。请注意，在最近邻分类器中异常数据点生成可能不正确的预测区域（例如蓝色区域中的绿点），而5-近邻分类器抚平了这些不正确的地方，可能会对测试数据产生更好的泛化（未示出）。</font>
- - -

在实践中，你将几乎总是想用k近邻。但是，你应该使用什么样的k值？我们下面介绍这个问题。

## 超参数调整的验证集
k-近邻分类器需要对设置k的值。但是值为多少效果最好？此外，我们看到，我们可以使用有许多不同距离函数，：L1，L2，还有很多其他我们甚至没有考虑的选择（如：点积）。这些选择被称为超参数，在设计很多机器学习算法时它们都会经常出现。但是选择什么样的值通常并不清楚。

你也许会认为，我们应该尝试许多不同的值，看看哪个效果最好。这是一个很好的想法，这确实是我们会做的，但这一定要非常谨慎。特别是，**不能用测试集调整超参数**。无论何时你设计机器学习算法，你应该把测试集当做非常宝贵的资源，应该树立只有在最后时刻才能使用它们的观念。否则，真正的危险是你可能在测试集上把你的超参数调整的很好，但一旦你部署你的模型，你会看到性能显著降低。在实践中，我们会说，你的测试集**过拟合**。另外一个看待这个问题的角度是，如果你在测试集上调整你的超参数，你实际上是把测试集当做训练集在使用，因此你获得的性能同你部署模型后真正观察到的性能相比会过于乐观。但是，如果你只在最后使用一次测试集，它仍然是能非常好的衡量你的分类器泛化能力（稍后我们会看到更多关于泛化能力的讨论）。
*`Evaluate on the test set only a single time, at the very end.`*
幸运的是，这有调整超参数的正确方式，并且不会触及测试集。想法是把训练集分成两部分：略小的训练集和我们所说的验证集。以CIFAR-10为例，我们可以使用训练集中49,000个图像进行训练，并留下1000个图像作为验证集。这个验证集基本上是一个用来调整超参数假的测试集。

以下是在CIFAR-10这个例子中用法：

```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```
这段程序结束后，我们可以绘制曲线图，显示其中效果最好的K值。然后，我们会一直使用这个值，并且在真正的测试集上评估一次。

*`把训练集分割成训练集和验证集。使用验证集调整所有的超参数。最后在测试集上运行一次并报告性能。`*

**交叉验证** 在训练数据集（因此也是验证数据集）规模较小的情况下，人们有时会使用更加先进的方法进行超参数调整，这种方法被称为*交叉验证*。继续前面的例子，这个想法是，为了避免随意挑选1,000个数据点作为验证集和余下的作为训练集，你可以通过遍历确定K值的不同验证集得到平均的性能，这样可以得到更好的更低噪音的估值。例如，在5次交叉验证中，我们将在训练数据分成5等份，使用其中份进行训练，1份进行验证。然后，我们将依次将每个等份作为验证集，评估性能，最后求不同等份性能的平均值。

- - -
![cvplot](http://cs231n.github.io/assets/cvplot.png)
<font size=2>参数k为5时，5倍交叉验证运行例子。Example of a 5-fold cross-validation run for the parameter k. 对于每个k值，我们在训练4倍和评估5日。因此，对于每K个收到5精度上验证倍（精度为y轴，每个结果是一个点）。For each value of k we train on 4 folds and evaluate on the 5th. Hence, for each k we receive 5 accuracies on the validation fold (accuracy is the y-axis, each result is a point). The trend line is drawn through the average of the results for each k and the error bars indicate the standard deviation. Note that in this particular case, the cross-validation suggests that a value of about k = 7 works best on this particular dataset (corresponding to the peak in the plot). If we used more than 5 folds, we might expect to see a smoother (i.e. less noisy) curve.</font>
- - -
**实践** 在实践中，人们更喜欢用一次验证，而不是交叉验证，因为交叉验证的运算量非常大。人们倾向于使用50％-90％的训练数据进行训练，剩余的验证。然而，这取决于多种因素：例如，如果超参数的数目大则可能更喜欢使用更大验证集。如果在验证集合样本的数目小（可能只有几百个左右），使用交叉验证则更加安全。实践中典型的交叉验证数目是3倍，5倍或10倍交叉验证。
- - -
![crossval](http://cs231n.github.io/assets/crossval.jpeg)
<font size=2>常用数据集。列出了训练和测试集。训练集被分成若干等份（例如这里5份）。1-4等份成为训练集。一个等份（如这里黄色的第5等份）作为验证集并且被用于调整超参数。交叉验证会分别从1-5等份挑选验证集。这被称为5-倍交叉验证。最后一旦训练好模型，确定好所有超参数，就用测试数据（红色）对该模型进行一次评价。</font>
- - -

**最近邻分类的优缺点** 最近邻分类的优缺点值得思考。显然，一个优点是，很容易实现和理解。此外，这个分类器不花时间来训练，因为所需要的就是存储训练数据，有可能需要索引训练数据。但是，我们在测试时会付出计算成本，因为把测试集分类时需要和每一个训练样本进行比对。这是退步，因为在实践中，我们经常更加关心的测试的效率而不是训练时的效率。事实上，我们在后面课程中会涉及的深度神经网络把这个平衡推向了另一个极端：训练时代价很高，但是一旦训练结束后把一个新的测试样本分类就没有什么代价了。这种操作方式在实践中更加可取。

顺便说一句，最近邻分类器的计算复杂度是一个活跃的研究领域，并且几个近似最近邻（ANN）的算法和库存在，可以加速最近邻在数据集的查询（例如[FLANN](http://www.cs.ubc.ca/research/flann/)）。这些算法允许折中最邻近算法的正确率和算法的时空复杂度，算法通常依赖于预处理/索引阶段，涉及构建kd树，或运行k-means算法。最近邻分类有时在某些情况下（特别是如果数据是低维的）是一个不错的选择，但很少适合于图像分类问题。一个原因是图像是高维的对象（即它们通常含有许多像素），以及高维空间的距离是非常反直觉的。下图说明了这一点，我们上面开发的基于像素的L2相似度和感官的相似度有很大不同：

- - -
![samenorm](http://cs231n.github.io/assets/samenorm.png)
<font size=2>高维数据（尤其是图像）中基于像素的距离非常的不直观。原始图像（左）和它旁边三个图像的L2距离非常远。显然，逐像素距离在感官或语义相似度上根本不适用。</font>
- - -

下面是多个图像来说服大家，用像素差异比较图像是不够的。我们可以使用一个名为[t-SNE](http://lvdmaaten.github.io/tsne/)可视化技术处理CIFAR-10图像并将其嵌入二维中，and embed them in two dimensions so that their (local) pairwise distances are best preserved. 在该可视化中，显示出来相近的图像在上述我们开发出来的L2距离上非常接近：
- - -
![pixels_embed_cifar10](http://cs231n.github.io/assets/pixels_embed_cifar10.jpg)
<font size=2>使用t-SNE将CIFAR-10图像以二维方式展示。这幅图片中相邻的图片在L2像素距离上也是相近的。请注意背景的影响比真实的意义大的多。点击[这里](http://cs231n.github.io/assets/pixels_embed_cifar10_big.jpg)查看一个更大的可视化版本。</font>
- - -
具体地，注意，互相接近的图片随着颜色或者背景不同而变动，而不是它们真实的意义。例如，可以看到一条狗非常靠近一个青蛙因为两者正好是白色背景。理想情况下，我们希望所有的10个类别的图像形成自己的集群，使同一类的图片都在附近彼此不分，而与其他的特点和变化（如背景）无关。然而，要达到这个性能，我们将必须超越原始像素。

## 总结

* 我们介绍了**图像分类**问题，这里我们有一组都标有一个单一类别的图像。我们接着预测新的测试图像的类别，并测量预测的准确性。
* 我们引入了一个简单的分类器叫做**最近邻分类器**。我们看到，和这个分类器相关的多个超参数（如k的值，或例子中用来比较的不同类型的距离），没有明显的办法选择他们。。
* 我们看到，正确设置这些超参数的方法是将训练数据分为两个：一个训练集和一个我们称之为验证集的假的测试集。我们尝试不同的超参数值，并保存在验证集上有最佳性能的值。
* 如果有缺乏训练数据的问题，我们讨论了一个叫做**交叉验证**的程序，它可以在评估超参数时帮助减少噪音。
* 一旦发现最佳超参数，我们解决了这些问题，并在实际的测试集执行一次评估。
* 我们看到，最近邻分类可以在CIFAR-10上得到约40％的精确度。这是简单的实现，但是要求我们存储整个训练集，并且在测试集评估时消耗非常大。
* 最后，我们看到，在原始像素上使用L1或L2距离并不合适，因为距离和背景、图像的颜色分布的相关性比它们的实际内容更强。

在接下来的课程，我们将着手解决这些挑战，并获得最终解决方案，可以达到90％的精确度，可以让我们在一次训练完成之后就彻底放弃训练集，使得我们能够在不到一毫秒内评估一个测试图像。







- - -
<font size=2>博客地址：[52ml.me](http://www.52ml.me)
原创文章，版权声明：自由转载-非商用-非衍生-保持署名 | [Creative Commons BY-NC-ND 3.0](http://creativecommons.org/licenses/by-nc-nd/3.0/deed.zh)</font>
- - -