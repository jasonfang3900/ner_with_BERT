import numpy as np

import torch
import joblib # 磁盘缓存等功能

import config
import dataset
import engine
from model import EntityModel
import os
import re

def map_BERTtoken2usual(tokenized_sentence):
    tokenizer = config.TOKENIZER
    token_spans = []
    usual_tokens = []
    is_substr = False
    for token_loc, token_id in enumerate(tokenized_sentence):
        bert_token = tokenizer._convert_id_to_token(token_id)
        if re.match("##",bert_token):
            substr = bert_token.replace("##", "", 1)
            usual_tokens[token_spans_len -1] += substr
            token_spans[token_spans_len - 1].append(token_loc)
        else:
            usual_tokens.append(bert_token)
            token_spans.append([token_loc])
        token_spans_len = len(token_spans)
    return usual_tokens, token_spans
   
def align_labels2tokens(token_spans, predict_labels):
    aligned_labels = []
    for span in token_spans:
        aligned_labels.append(predict_labels[span[0]])
    return aligned_labels

def main():
    
    
    meta_data = joblib.load("../data/meta.bin") # 加载meta_data
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]
    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))


    sentence = "Jason is going to China tomorrow afternoon."

    tokenized_sentence = config.TOKENIZER.encode(
        sentence,
        # add_special_tokens=False
    )
    usual_tokens, token_spans = map_BERTtoken2usual(tokenized_sentence)

    sentence = sentence.split(" ")
    print("sentence: ")
    print(sentence)
    print(usual_tokens)
    print("*" * 50)

    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        pos=[[0] * len(sentence)], 
        tags=[[0] * len(sentence)]
    )


    device = torch.device("cuda")
    model = EntityModel(num_pos, num_tag)
    model.load_state_dict(torch.load(os.path.join(config.OUTPUT_MODEL_PATH, "best.pt")))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _ = model(**data)
        # tag 的size为[1, 128, 17]，128为最大句长，17为标签类别数
        raw_pre_tag = tag.argmax(2).cpu().numpy().reshape(-1)[:len(tokenized_sentence)]
        pre_tag = align_labels2tokens(
                token_spans,
                enc_tag.inverse_transform(raw_pre_tag)
                )

        raw_pre_pos = pos.argmax(2).cpu().numpy().reshape(-1)[:len(tokenized_sentence)]
        
        pre_pos = align_labels2tokens(
                token_spans,
                enc_pos.inverse_transform(raw_pre_pos)
                )

        print("token\t词性\t命名实体")
        print("*" * 50)
        for i in range(len(pre_pos)):
            print("%s\t%s\t%s\t" %(usual_tokens[i], pre_tag[i], pre_pos[i]))
if __name__ == "__main__":
    main()
