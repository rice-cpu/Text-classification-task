# 基于BERT的自定义中文文本分类微调

本项目基于PyTorch与Hugging Face`transformers`库，实现了针对自定义中文数据集的标签序列分类微调流水线。

## 1.项目组成

项目严格遵循模块化高内聚、低耦合的解耦规范，各组件职责对账如下：
* `config.json`：全局动态调控。集中管理 `batch_size`、`learning_rate`、`max_length` 等所有超参数。
* `dataset.py`：数据加载处理。继承PyTorch `Dataset`，负责文本读取与截断。
* `model.py`：模型板块。构建自定义文本分类模型`berttextClassifier`。
* `train.py`：训练循环。负责反向传播、梯度裁剪、验证集检验及SwanLab。
* `main.py`：总入口。实例化，挂载官方 `DataCollatorWithPadding` 动态打包和训练。
* `inference.py`：测试。根据训练的模型权重，对单行文本进行测验。
* `checkpoints/`：保存权重和配置文件。用于存放历史最优评估得分的完整配置（`best_checkpoint.pt`）。


## 2.本地数据格式说明

脚本支持读取本地自定义 `.txt` 文本数据集。每一行物理数据均采用行业标准的固定分隔符 `_!_` 进行结构化解耦：



## 3.训练轨迹可视化（SwanLab 黄金曲线）
以下为调参优化后的曲线：

!![img.png](image/img.png)