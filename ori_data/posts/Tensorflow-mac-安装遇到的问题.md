---
title: Tensorflow mac 安装遇到的问题
date: 2017-02-20 19:03:49
tags:
- tensorflow
---
tensorflow 使用conda forge源安装的步骤是：
```
conda create -n tensorflow python=3.5
source activate tensorflow
conda install pandas matplotlib jupyter notebook scipy scikit-learn
conda install -c conda-forge tensorflow
```
<!---more--->
最后一步时会安装protobuf，由于网络问题导致安装失败，出现错误：

```
CondaError: CondaHTTPError: HTTP None None for url <None>
Elapsed: None

An HTTP error occurred when trying to retrieve this URL.
ConnectionError(ReadTimeoutError("HTTPSConnectionPool(host='binstar-cio-packages-prod.s3.amazonaws.com', port=443): Read timed out.",),)
```

解决办法参考[《tensorflow在linux系统上的安装》](http://blog.csdn.net/zhangweijiqn/article/details/53199553)
具体方法：

1. 分别到[protobuf](https://anaconda.org/conda-forge/protobuf/files)和[tensorflow](https://anaconda.org/conda-forge/tensorflow/files)下载对应版本的安装包。
2. 将下载的安装包放到anaconda/pkgs这个目录下，需要重命名。
3. 用conda install path_to_file直接安装。

---
博客地址：[52ml.me](http://www.52ml.me)<br>
原创文章，版权声明：自由转载-非商用-非衍生-保持署名 | [Creative Commons BY-NC-ND 3.0](http://creativecommons.org/licenses/by-nc-nd/3.0/deed.zh)
<br>
---

