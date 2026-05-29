# 数据集读取说明

本目录按照 CLOVA-tool 的风格放置数据集读取器。真实数据仍然放在项目根目录的 `data/` 下，`Datasets/` 只负责把标注文件整理成 MoReVQA 可以逐条推理的样本。

## NExT-QA

当前已实现 `NextQADataset`，支持下面这种目录：

```text
data/nextqa/
  annotations/
    map_vid_vidorID.json
    train.csv
    val.csv
    test.csv
  videos/
    0000/
      2440175990.mp4
      ...
```

读取逻辑：

1. 从 `annotations/{split}.csv` 读取 `video, question, answer, a0...a4`。
2. 用 `annotations/map_vid_vidorID.json` 把 `video` 映射到 `videos/<子目录>/<video_id>.mp4`。
3. 返回字段包括 `video_path`、`question`、`options`、`answer_index`、`answer`、`qid` 和 `type`。
4. 用 `data_num` 控制请求数量，用 `start_index` 和 `interval` 控制从哪一行开始、每隔几行取一次。

在 Python 中使用：

```python
from Datasets.loaders import NextQADataset, build_nextqa_dataloader

dataset = NextQADataset(
    data_root="data/nextqa",
    split="val",
    data_num=10,
    start_index=0,
    interval=1,
)
dataloader = build_nextqa_dataloader(dataset, batch_size=1, shuffle=False)
```

更推荐直接运行根目录的 `nextqa_demo.py`，它已经完成了 DataLoader、MoReVQA pipeline、结果保存和简单准确率统计。
