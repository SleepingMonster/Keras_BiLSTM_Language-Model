# Keras_BiLSTM_Language-Model
中山大学 自然语言处理项目：中文语言模型。

Keras实现，BiLSTM框架。

## Readme

#### 实验环境

- `keras` 2.3.1版本和`tensorflow` 2.2版本（或者其他相匹配的`keras`和`tensorflow`版本）

- `keras_contrib`库、`gensim`库、`pickle`库、`tqdm`库

#### 实验工具

jupyter notebook

#### 文件组织

- 词向量：`sgns.wiki.word` 文件
  - 来自于 https://github.com/Embedding/Chinese-Word-Vectors 中“Various Domains”中的“Word”的"Wikipedia_zh 中文维基百科"。
- `task2`文件：
  - `task2.ipynb`（jupyter notebook格式）和`task2.py`，建议执行`task2.ipynb`。
  - 逐块执行即可得到训练`EPOCH`次数之后的测试集的预测结果，并会输出预测结果。
    - `EPOCH`的取值可在`train`函数的参数中进行调整。
  - 结果：训练集迭代运行25次之后的结果保存为`result.txt`文件。
