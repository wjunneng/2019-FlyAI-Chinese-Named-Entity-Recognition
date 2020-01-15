# 2019-FlyAI-Chinese-Named-Entity-Recognition
2019 FlyAI 中文的命名实体识别

## 待提升的方案

### 
    1、引入伪标签
    2、引入标签平滑（多类别交叉熵的情况）


###
    当网络的评价指标不在提升的时候，可以通过降低网络的学习率来提高网络性能:
    optimer指的是网络的优化器
    mode (str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
    factor 学习率每次降低多少，new_lr = old_lr * factor
    patience=10，容忍网路的性能不提升的次数，高于这个次数就降低学习率
    verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
    threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
    cooldown(int)： 冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
    min_lr(float or list):学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
    eps(float):学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
    
    注意：
        1.在模型中有BN层或者dropout层时，在训练阶段和测试阶段必须显式指定train()
            和eval()。
        2.一般来说，在验证或者是测试阶段，因为只是需要跑个前向传播(forward)就足够了，
            因此不需要保存变量的梯度。保存梯度是需要额外显存或者内存进行保存的，占用了空间，
            有时候还会在验证阶段导致OOM(Out Of Memory)错误，因此我们在验证和测试阶段，最好显式地取消掉模型变量的梯度。
            使用torch.no_grad()这个上下文管理器就可以了。
        3.weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度， 所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。
###