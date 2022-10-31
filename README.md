本项目基于TensorFlow 1.14版本开发，主要改进为：在参数间和计算节点间两个维度均加入优先级调度，并将这两个维度统一起来为通信数据流分配优先级。该版本仅支持`verbs`通信方式。


## 安装

训练框架安装同TensorFlow

优先级分配策略优化基于 `ParaMapModule.py` 脚本，需安装相应的python依赖包，详见requirements.txt。

## 使用

### 配置方式

使用方式基本与TensorFlow相似。此外，需要定义以下环境变量以配置优先级调度。

* `FLOW_PRIORITY`: 使用的优先级标签，当不指定时，默认不使用优先级调度。当使用多个优先级队列时，可以指定多个标签，如`FLOW_PRIORITY=64,96`。
* `MAPPING_MODE`: 优先级分配策略。`0`: 基于功能划分，卷积层使用高优先级，全连接层使用低优先级；`1`: 均匀划分，将所有层在使用的优先级上进行均分；`2`：指定划分，通过指定`PRIORITY_THRESHOLD`来指定优先级划分阈值；`3`：自动划分，基于贝叶斯优化算法，对优先级分配结果自动进行迭代优化。
* `PRIORITY_THRESHOLD`: 优先级划分阈值，仅当使用`指定划分`方式时有效。以使用3个优先级时配置`PRIORITY_THRESHOLD=0.4,0.7`为例，前40%的层使用最高优先级，后30%的层使用最低优先级，其余中间层使用次高优先级。
* `SLICING_THRESHOLD`：切片阈值，基本单位为一个float参数。
* `PARTITIONER`：参数切分份数，如`PARTITIONER=8`可以将所有参数分成8份。
* `DIST_STRAG`：是否区分straggler, `0`：不区分，`1`：区分。
* `PERF_GAP`: 计算节点性能离散化阈值，`PERF_GAP=10`表示10%。

当使用`自动划分`策略时，需要`ParaMapModule.py` 脚本的配合。其作为一个单独的进程与TensorFlow进行socket通信，可通过`-s`选项指定通信对端地址，这里的通信对端指`worker 0`，其他选项详见该脚本文件。

### 示例：

训练端：

`MAPPING_MODE=3 PRIORITY_THRESHOLD=0.4,0.7, DIST_STRAG=1 PERF_GAP=10 FLOW_PRIORITY=64,96SLICING_THRESHOLD=1024000 PARTITIONER=8 python train.py`

优先级分配优化端：
`python ParaMapModule.py -s 192.168.1.10`
