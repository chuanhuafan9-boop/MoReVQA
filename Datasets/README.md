# 数据集适配目录

本目录与 CLOVA-tool 的 `Datasets/` 职责对应，用于后续添加数据集读取器、批量推理器和评测器。

实际视频与标注文件继续存放在项目根目录的 `data/` 中，以保持已经创建好的目录不变：

```text
data/
  nextqa/{videos,annotations}
  ivqa/{videos,annotations}
  egoschema/{videos,annotations}
  activitynet_qa/{videos,annotations}
  next_gqa/{videos,annotations}
  activitynet_paragraphs/{videos,annotations}
```

当前实现支持从这些目录选取单个视频问题运行；论文全量评测的 dataset loader 与 evaluator 尚待实现。
