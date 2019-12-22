# -*- coding: utf-8 -*
"""
实现模型的调用
"""
from flyai.dataset import Dataset
from model import Model

data = Dataset()
model = Model(data)
# p = model.predict(
#     source="新华社 北京 5 月 8 日 电 中国 跳水 名将 余卓成 7 日 在 美国 佛罗里达州 举行 的 国际泳联 跳水 大奖赛 上 ， 获得 男子 一米板 冠军 。 新华社 北京 5 月 8 日 电 中国 跳水 名将 余卓成 7 日 在 美国 佛罗里达州 举行 的 国际泳联 跳水 大奖赛 上 ， 获得 男子 一米板 冠军 。 新华社 北京 5 月 8 日 电 中国 跳水 名将 余卓成 7 日 在 美国 佛罗里达州 举行 的 国际泳联 跳水 大奖赛 上 ， 获得 男子 一米板 冠军 。 新华社 北京 5 月 8 日 电 中国 跳水 名将 余卓成 7 日 在 美国 佛罗里达州 举行 的 国际泳联 跳水 大奖赛 上 ， 获得 男子 一米板 冠军 。 新华社 北京 5 月 8 日 电 中国 跳水 名将 余卓成 7 日 在 美国 佛罗里达州 举行 的 国际泳联 跳水 大奖赛 上 ， 获得 男子 一米板 冠军 。 新华社 北京 5 月 8 日 电 中国 跳水 名将 余卓成 7 日 在 美国 佛罗里达州 举行 的 国际泳联 跳水 大奖赛 上 ， 获得 男子 一米板 冠军 。")
# before = "新华社  北京"
# print(len(before.split(' ')))
# p = model.predict(source=before)
p = model.predict_all(
    [{"source": "据 新华社 地拉那 6 月 15 日 电 阿尔巴尼亚 总理 纳诺 15 日 在 会见 来访 的 瑞典 国防 大臣 叙多 时 表示 ， 北约 在 巴尔干 地区 举行 空军 演习 时 ， 阿尔巴尼亚 可以 向 北约 提供 包括 机场 在内 的 所有 军事 后勤 设施 。"},
     {"source": "在 闭幕 会议 上 ， 联合国 副 秘书长 兼 联合国 国际 禁毒署 署长 阿拉 奇 代表 联合国 秘书长 安南 致 闭幕词 。"},
     {
         "source": "在 闭幕 会议 上 ， 联合国 副 秘书长 兼 联合国 国际 禁毒署 署长 阿拉 奇 代表 联合国 秘书长 安南 致 闭幕词 。 在 闭幕 会议 上 ， 联合国 副 秘书长 兼 联合国 国际 禁毒署 署长 阿拉 奇 代表 联合国 秘书长 安南 致 闭幕词 。 在 闭幕 会议 上 ， 联合国 副 秘书长 兼 联合国 国际 禁毒署 署长 阿拉 奇 代表 联合国 秘书长 安南 致 闭幕词 。 在 闭幕 会议 上 ， 联合国 副 秘书长 兼 联合国 国际 禁毒署 署长 阿拉 奇 代表 联合国 秘书长 安南 致 闭幕词 。 在 闭幕 会议 上 ， 联合国 副 秘书长 兼 联合国 国际 禁毒署 署长 阿拉 奇 代表 联合国 秘书长 安南 致 闭幕词 。 在 闭幕 会议 上 ， 联合国 副 秘书长 兼 联合国 国际 禁毒署 署长 阿拉 奇 代表 联合国 秘书长 安南 致 闭幕词 。 在 闭幕 会议 上 ， 联合国 副 秘书长 兼 联合国 国际 禁毒署 署长 阿拉 奇 代表 联合国 秘书长 安南 致 闭幕词 。 在 闭幕 会议 上 ， 联合国 副 秘书长 兼 联合国 国际 禁毒署 署长 阿拉 奇 代表 联合国 秘书长 安南 致 闭幕词 。 在 闭幕 会议 上 ， 联合国 副 秘书长 兼 联合国 国际 禁毒署 署长 阿拉 奇 代表 联合国 秘书长 安南 致 闭幕词 。 在 闭幕 会议"},
     {"source": "强令 违章 冒险 作业 罪"}
     ])

print([len(i) for i in p])
print(p)
