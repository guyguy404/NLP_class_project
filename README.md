# 运行说明

## 训练

```bash
python scripts/main.py --device 0 --model_name bert --bert_name bert-base-chinese
```

### 参数说明

- `model_name`: 可选择 `baseline`, `bert`, `MISCA_Att`, `MISCA_noAtt`
- `bert_name`: 用于选择预训练的 BERT 模型种类，可选择 `bert-base-chinese`, `chinese-bert-wwm-ext`



## 测试

```bash
python scripts/main.py --device 0 --model_name bert --bert_name bert-base-chinese --testing
```



### 预训练模型

+ Bert: https://huggingface.co/bert-base-chinese，下载后放置于 `/model/bert-base-chinese`
+ Bert-WWM: https://huggingface.co/hfl/chinese-bert-wwm-ext，下载后放置于 `/model/chinese-bert-wwm-ext`

