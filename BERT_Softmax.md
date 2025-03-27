# Bert-Softmax

- 论文名称：
- 论文地址：
- Github 代码：

## 一、前言

### 1.1 介绍

基于Pytorch的 BERT+Softmax 命名实体识别。

### 1.2 requirement

1. 1.1.0 =< PyTorch < 1.5.0
2. cuda=9.0
3. python3.6+

## 二、环境搭建

### 2.1 下载代码 

```s
    $ git clone 
```

### 2.2 构建环境

```s
    $ conda create -n py38 python=3.8       # 创建新环境
    $ source activate py38                   # 激活环境
```

### 2.3 安装依赖 

```s
    $ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 三、 训练数据介绍

### 3.1 dataset list

1. cner: datasets/cner
2. CLUENER: https://github.com/CLUEbenchmark/CLUENER

### 3.2 input format

Input format (prefer BIOS tag scheme), with each character its label for one line. Sentences are splited with a null line.

```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER

我	O
跟	O
他	O
```

## 四、模型微调

### 4.1 模型训练参数修改

训练之前，需要修改 run_ner_softmax.sh 参数

```s
CURRENT_DIR=`pwd`                                  # 数据源根目录
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cner"                                   # 模型处理任务

python run_ner_softmax.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --loss_type=ce \                                 # loss 函数     
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=3.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
```

> 注：这里 根据 所用机器 算力 大小 调整 以下参数 
>    train_max_seq_length：      训练数据长度
>    per_gpu_train_batch_size：  训练 batch size 大小

### 4.2 模型训练

```s
    $ bash run_ner_softmax.sh
```

## 五、模型微调实验

### 5.1 CLUENER result

The overall performance of BERT on **dev**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.7897     | 0.8031     | 0.7963    |

### 5.2 Cner result

The overall performance of BERT on **dev(test)**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.9586(0.9566)     | 0.9644(0.9613)     | 0.9615(0.9590)     |

## 致谢

1. [Github:transformers] https://github.com/huggingface/transformers
2. [Paper:Bert] https://arxiv.org/abs/1810.04805
3. [Paper:RDrop] https://arxiv.org/abs/2106.14448
