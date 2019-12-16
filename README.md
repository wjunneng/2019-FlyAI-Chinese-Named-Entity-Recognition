# 2019-FlyAI-Chinese-Named-Entity-Recognition
2019 FlyAI 中文的命名实体识别

## 待提升的方案

### 1、 同时训练训练集和验证集.
### 2、 为每条数据开头添加一个字符source->‘[UNK]’,target->‘O’; 然后,在预测的时候进行截断,查看效果是否更好.