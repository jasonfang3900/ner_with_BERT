import pandas as pd
import numpy as np

import torch

import joblib # 磁盘缓存等功能
from sklearn import preprocessing
from sklearn import model_selection

# 创建一个schedule, 学习率先从0到从优化器的初始lr线性增大，再从lr到0线性减小。
from transformers import get_linear_schedule_with_warmup
# Adam algorithm with weight decay fix
from transformers import AdamW

import config
import dataset
import engine
from model import EntityModel
import os

def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    # 去除NaN值，替换为sentence id
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
    enc_pos= preprocessing.LabelEncoder()
    enc_tag= preprocessing.LabelEncoder()

    # 将标签分配一个0—n_classes-1之间的编码数
    df.loc[:, "POS"] =enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] =enc_tag.fit_transform(df["Tag"])
    # 返回一个numpy.ndarray，其中的每个元素是一个list，即一个句子，list的每个元素是一个词。
    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, pos, tag, enc_pos, enc_tag

def main():
    sentences, pos, tag, enc_pos, enc_tag = process_data(config.TRAINING_FILE)
    
    meta_data = {
        "enc_pos": enc_pos,
        "enc_tag": enc_tag
    }
    joblib.dump(meta_data, "../data/meta.bin") # 保存mata_data
    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    # 按9/1的比例划分训练集和测试集
    (
        train_sents,
        test_sents,
        train_pos,
        test_pos,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.1)

    train_dataset = dataset.EntityDataset(
        texts=train_sents, pos=train_pos, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=8
    )

    valid_dataset = dataset.EntityDataset(
        texts=test_sents, pos=test_pos, tags=test_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda")
    model = EntityModel(num_pos, num_tag)
    model.to(device)

    # named_parameters()方法返回的是一个生成器，每个元素是一个元组，包含参数名和参数值
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayNorm.bias", "LayerNorm.weight"]
    # any，可迭代对象中只要有一个元素为True就返回True，如果全是False则返回False
    optimizer_parameters = [
        {
            "params": [
                param for name, param in param_optimizer if not any(nd in name for nd in no_decay)
            ],
            "weight_decay": 1e-3
        },
        {
            "params": [
                # no_decay列表中的参数不使用weight_decay
                param for name, param in param_optimizer if any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0
        }
    ]

    # 迭代次数
    num_train_steps = int(len(train_sents) / config.TRAIN_BATCH_SIZE * config.EPOCHS )
    # 优化器
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    # 学习率调整
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = engine.eval_fn(valid_data_loader, model, optimizer, device)
        print(f"train loss: {train_loss}")
        print(f"valid loss: {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_MODEL_PATH, "best.pt"))
            best_loss = test_loss

    

if __name__ == "__main__":
    main()



