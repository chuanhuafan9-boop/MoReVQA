# /home/fanchuanhua/project/MoReVQA  整个项目在服务器里面的地址
# MoReVQA Project README

本项目基于 CVPR 2024 论文 **MoReVQA: Exploring Modular Reasoning Models for Video Question Answering** 实现，是一个面向视频问答任务的模块化推理框架。项目核心思想是让大语言模型先把视频问题解析成可执行的工具调用程序，再结合视觉语言模型、目标定位模型和图文匹配模型完成视频问答。

当前项目已经按照 CLOVA-tool 的结构方式整理，只保留 **MoReVQA pipeline**，不再保留 JCEF、LLM-only 等 baseline 代码。当前主要支持 NExT-QA 数据集批量运行，也支持单个视频问题运行。

## 1. 项目核心流程

MoReVQA 的主流程分为四步：

1. M1 Event Parsing：由 LLM 解析问题中的事件、时序关系、问题类型和 OCR 需求。
2. M2 Grounding：根据 M1 的结果定位视频中和问题相关的帧。
3. M3 Reasoning：对上下文帧生成 caption，并对关键帧执行 VQA 子问题推理。
4. Final Prediction：由 LLM 根据外部记忆中的所有中间结果输出最终答案。

主要入口在：

```text
framework/morevqa.py
```

其中 `MoReVQA.inference()` 是统一推理入口，调用顺序为：

```text
EventParsingStage -> GroundingStage -> ReasoningStage -> PredictionStage
```

对应论文中的：

```text
M1 -> M2 -> M3 -> Final Prediction
```

## 2. 模型对应关系

论文里的模型和本项目里的替代模型如下：

| 论文模型 | 论文作用 | 本项目替代模型 | 配置位置 |
| --- | --- | --- | --- |
| PaLM-2 | LLM，负责程序生成和最终预测 | Qwen3-30B-A3B-Instruct-2507 | `llm` |
| PaLI-3 (5B) | VLM，负责 caption、VQA、OCR | PaliGemma2-10B-Mix-448 | `vision_language` / `ocr` |
| OWL-ViT | 开放词汇目标定位 | google/owlvit-base-patch32 | `grounding.detector` |
| CLIP RN50 | 图文相似度匹配 | OpenAI CLIP RN50 | `grounding.image_text_scorer` |

注意：

```text
PaLM-2 和 PaLI-3 是两个不同模型。
PaLM-2 是 LLM。
PaLI-3 是 VLM。
```

本项目中：

```text
Qwen3 替代 PaLM-2。
PaliGemma2 替代 PaLI-3。
```

## 3. 目录结构

```text
.
├── configs/
│   └── LLM_config.yaml              # LLM、VLM、grounding、video、pipeline 配置
├── Datasets/
│   ├── loaders.py                   # NExT-QA 数据集读取器
│   ├── README.md                    # 数据集读取说明
│   └── __init__.py
├── data/
│   └── nextqa/
│       ├── annotations/
│       │   ├── map_vid_vidorID.json
│       │   ├── train.csv
│       │   ├── val.csv
│       │   └── test.csv
│       └── videos/
│           └── xxxx/
│               └── video_id.mp4
├── model/
│   ├── paligemma2-10b-mix-448/      # PaliGemma2 本地模型
│   ├── owlvit-base-patch32/         # OWL-ViT 本地模型
│   └── clip-rn50/
│       └── RN50.pt                  # OpenAI CLIP RN50 权重
├── engine/
│   ├── config.py                    # YAML 配置读取
│   ├── memory.py                    # MoReVQA 外部记忆结构
│   ├── schemas.py                   # 共享数据结构
│   ├── utils.py                     # ProgramGenerator、ProgramInterpreter、日志格式化
│   ├── video.py                     # 视频抽帧工具
│   └── models/
│       ├── adapters.py              # Qwen、PaliGemma2、OWL-ViT、CLIP 调用适配器
│       ├── registry.py              # 根据配置构建模型组件
│       └── base.py                  # 模型接口定义
├── framework/
│   ├── morevqa.py                   # MoReVQA 主框架
│   └── stages/
│       ├── event_parser.py          # M1 Event Parsing
│       ├── grounding.py             # M2 Grounding
│       ├── reasoning.py             # M3 Reasoning
│       └── prediction.py            # Final Prediction
├── prompts/
│   └── prompt_engineering.py        # M1/M2/M3/Final Prediction prompt
├── tools/
│   ├── event.py                     # M1 可调用工具
│   ├── grounding.py                 # M2 可调用工具
│   └── reasoning.py                 # M3 可调用工具
├── outputs/
│   ├── nextqa_val_predictions.jsonl # NExT-QA 批量结果
│   └── nextqa_val_traces/           # 每条样本的完整 trace
├── nextqa_demo.py                   # NExT-QA 批量运行入口
├── videoqa_demo.py                  # 单视频问答入口
├── main.py                          # 兼容入口，转发到 videoqa_demo.py
├── morevqa/
│   └── cli.py                       # 命令行入口
├── tests/
│   ├── check_dashscope_llm.py       # 单独检查百炼 Qwen API 连通性
│   └── test_*.py
├── requirements.txt
└── README.md
```

## 4. 环境安装

建议创建单独环境：

```bash
conda create -n morevqa python=3.10 -y
conda activate morevqa
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果你想把本项目作为本地包安装：

```bash
pip install -e . --no-deps
```

当前 `requirements.txt` 中包含：

```text
torch
torchvision
transformers
accelerate
sentencepiece
openai
httpx
opencv-python
Pillow
PyYAML
OpenAI CLIP
```

如果 `git+https://github.com/openai/CLIP.git` 下载失败，可以先确认服务器代理和 GitHub 访问是否正常。

## 5. 重要配置文件

主配置文件是：

```text
configs/LLM_config.yaml
```

这个文件控制 LLM、VLM、grounding、OCR、视频抽帧和 pipeline 参数。

### 5.1 `llm`

替换自己的 api-key
`llm` 部分负责调用 Qwen，替代论文中的 PaLM-2。

示例：

```yaml
llm:
  provider: openai_compatible
  client: openai
  api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: <your-dashscope-api-key>
  model: qwen3-30b-a3b-instruct-2507
  temperature: 0.0
  max_tokens: 1024
  timeout: 200
  proxy_url: null
  extra_body:
    enable_thinking: false
```

如果服务器访问百炼需要代理，则设置：

```yaml
proxy_url: http://127.0.0.1:7890
```

如果服务器直连百炼更稳定，则设置：

```yaml
proxy_url: null
```

注意：不要把真实 API Key 上传到公开仓库。建议使用环境变量保存 API Key，例如：

```bash
export DASHSCOPE_API_KEY="你的百炼API Key"
```

然后在配置中写：

```yaml
api_key: ${DASHSCOPE_API_KEY}
```

### 5.2 `vision_language`

`vision_language` 部分负责调用 PaliGemma2，替代论文中的 PaLI-3。

示例：

```yaml
vision_language:
  captioner:
    provider: paligemma2
    model_path: /home/fanchuanhua/project/MoReVQA/model/paligemma2-10b-mix-448
    prompt: caption en
    vqa_prefix: answer en
    temperature: 0.0
    max_new_tokens: 64
  vqa:
    provider: paligemma2
    model_path: /home/fanchuanhua/project/MoReVQA/model/paligemma2-10b-mix-448
    prompt: describe en
    vqa_prefix: answer en
    temperature: 0.0
    max_new_tokens: 64
```

`captioner.prompt` 用于上下文帧描述。

`vqa.vqa_prefix` 用于视觉问答。

### 5.3 `grounding`

`grounding` 部分负责 M2 阶段的视频帧定位。

```yaml
grounding:
  detector:
    provider: owlvit
    model_path: /home/fanchuanhua/project/MoReVQA/model/owlvit-base-patch32
    threshold: 0.12
    top_k: 8
    query_max_length: 16
  image_text_scorer:
    provider: openai_clip_rn50
    model_name: /home/fanchuanhua/project/MoReVQA/model/clip-rn50/RN50.pt
    threshold: 0.7
  verify_top_k: 8
  grounding_top_k: 6
  temporal_keep_ratio: 0.4
```

### 5.4 `video`

`video` 部分控制抽帧：

```yaml
video:
  sample_fps: 1.0
  max_frames: null
  context_frames: 16
```

论文中 MoReVQA 默认使用 16 个上下文帧。EgoSchema 任务通常使用 30 个上下文帧，但当前主要运行 NExT-QA，所以默认是 16。

## 6. 模型准备

模型建议放在项目根目录的 `model/` 下，每个模型一个子文件夹。

### 6.1 PaliGemma2

模型：

```text
google/paligemma2-10b-mix-448
```

目标目录：

```text
model/paligemma2-10b-mix-448/
```

下载示例：

```bash
huggingface-cli download google/paligemma2-10b-mix-448 \
  --local-dir model/paligemma2-10b-mix-448
```

如果 `huggingface-cli` 不可用：

```bash
python -m huggingface_hub.commands.huggingface_cli download google/paligemma2-10b-mix-448 \
  --local-dir model/paligemma2-10b-mix-448
```

### 6.2 OWL-ViT

模型：

```text
google/owlvit-base-patch32
```

目标目录：

```text
model/owlvit-base-patch32/
```

下载示例：

```bash
huggingface-cli download google/owlvit-base-patch32 \
  --local-dir model/owlvit-base-patch32
```

### 6.3 CLIP RN50

目标目录：

```text
model/clip-rn50/RN50.pt
```

如果配置中写：

```yaml
model_name: RN50
```

则 OpenAI CLIP 会自动下载到用户缓存目录。

如果你已经手动下载 `RN50.pt`，建议配置为：

```yaml
model_name: /home/fanchuanhua/project/MoReVQA/model/clip-rn50/RN50.pt
```

### 6.4 Qwen3

当前项目通过阿里云百炼 OpenAI-compatible API 调用 Qwen3，不需要本地下载 Qwen3 权重。

模型 code：

```text
qwen3-30b-a3b-instruct-2507
```

百炼接口：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 7. 数据集准备

当前主要支持 NExT-QA。

目录应为：

```text
data/nextqa/
├── annotations/
│   ├── map_vid_vidorID.json
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── videos/
    ├── 0000/
    │   ├── xxxxxxxxxx.mp4
    │   └── ...
    ├── 0001/
    └── ...
```

`Datasets/loaders.py` 中的 `NextQADataset` 会：

1. 读取 `annotations/{split}.csv`。
2. 根据 `map_vid_vidorID.json` 找到视频子目录。
3. 拼出真实视频路径。
4. 读取问题、候选答案和标准答案。
5. 返回给 MoReVQA pipeline。

NExT-QA 标注 CSV 的关键列：

```text
video, question, answer, qid, type, a0, a1, a2, a3, a4
```

其中 `answer` 是 0 到 4 的选项编号。

## 8. 主要运行脚本

### 8.1 NExT-QA 批量运行

入口：

```text
nextqa_demo.py
```

运行前主要修改文件顶部变量：

```python
SPLIT = "val"
REQUEST_NUM = 200
START_INDEX = 0
INTERVAL = 1
SKIP_MISSING_VIDEOS = True
FORCE_MOCK = False
```

含义：

```text
SPLIT                 使用 train / val / test 哪个划分
REQUEST_NUM           本次运行多少个有效样本；None 或 0 表示跑完整 split
START_INDEX           从 CSV 第几行开始，0 表示第一条
INTERVAL              每隔多少条取一个样本，1 表示连续读取
SKIP_MISSING_VIDEOS   找不到视频时是否跳过
FORCE_MOCK            是否使用 mock 模型快速验证流程
```

在 PyCharm 里运行：

```text
右键 nextqa_demo.py -> Run
```

在命令行运行，并保存日志：

```bash
CUDA_VISIBLE_DEVICES="0" nohup python nextqa_demo.py > nextqa_demo_0.log 2>&1 &

tail -f nextqa_demo_0.log
```

如果想指定第 1 张显卡：

```bash
CUDA_VISIBLE_DEVICES="1" nohup python nextqa_demo.py > nextqa_demo_1.log 2>&1 &
```

注意：`CUDA_VISIBLE_DEVICES="1"` 后，程序内部会把物理第 1 张卡看成 `cuda:0`。

### 8.2 命令行方式运行 NExT-QA

跑 val 集前 5 个样本：

```bash
CUDA_VISIBLE_DEVICES="0" nohup python -m morevqa.cli \
  --dataset nextqa \
  --config configs/LLM_config.yaml \
  --data-root data/nextqa \
  --split val \
  --num-samples 5 \
  --output outputs/nextqa_val_predictions.jsonl \
  --trace-dir outputs/nextqa_val_traces \
  > nextqa_val_0_5.log 2>&1 &
```

查看日志：

```bash
tail -f nextqa_val_0_5.log
```

从第 100 条开始，每隔 5 条取 1 条，总共跑 20 个样本：

```bash
CUDA_VISIBLE_DEVICES="0" nohup python -m morevqa.cli \
  --dataset nextqa \
  --config configs/LLM_config.yaml \
  --data-root data/nextqa \
  --split val \
  --start-index 100 \
  --interval 5 \
  --num-samples 20 \
  --output outputs/nextqa_val_sparse_predictions.jsonl \
  --trace-dir outputs/nextqa_val_sparse_traces \
  > nextqa_val_sparse.log 2>&1 &
```

### 8.3 单视频问答

入口：

```text
videoqa_demo.py
```

在文件顶部修改：

```python
VIDEO_PATH = PROJECT_ROOT / "examples" / "demo_video.mp4"
QUESTION = "What is happening in the video?"
OPTIONS = [
    "a colored square is moving across the screen",
    "the video is blank",
    "someone is reading text aloud",
]
FORCE_MOCK = False
```

运行：

```bash
python videoqa_demo.py
```

或命令行：

```bash
python -m morevqa.cli \
  --config configs/LLM_config.yaml \
  --video data/nextqa/videos/1106/4010069381.mp4 \
  --question "how do the two man play the instrument" \
  --options "roll the handle" "tap their feet" "strum the string" "hit with sticks" "pat with hand" \
  --trace outputs/single_trace.json
```

## 9. 输出文件说明

### 9.1 `outputs/nextqa_val_predictions.jsonl`

这个文件是批量运行结果汇总。一行是一个样本。

典型字段：

```json
{
  "sample_id": "val-4010069381-6",
  "video_id": "4010069381",
  "video_path": ".../data/nextqa/videos/1106/4010069381.mp4",
  "question": "how do the two man play the instrument",
  "options": ["roll the handle", "tap their feet", "strum the string", "hit with sticks", "pat with hand"],
  "answer_index": 0,
  "answer": "roll the handle",
  "prediction": "strum the string",
  "predicted_index": 2,
  "correct": false,
  "trace_path": ".../outputs/nextqa_val_traces/val-4010069381-6.json"
}
```

这个文件用于：

1. 统计准确率。
2. 查看每条问题的预测答案。
3. 筛选错误样本。
4. 找到对应的 trace 文件。

### 9.2 `outputs/nextqa_val_traces/*.json`

这个目录里每个 JSON 是单个样本的完整推理过程。

里面包含：

```text
prediction.answer          最终后处理答案
prediction.raw_response    Final Prediction LLM 原始回答
memory.question            原始问题
memory.options             候选答案
memory.frame_ids           抽帧后的帧编号
memory.working_question    M1 改写后的问题
memory.temporal_hint       M1 判断的时序提示
memory.conjunction         M1 判断的事件关系
memory.qa_type             问题类型
memory.require_ocr         是否需要 OCR
memory.event_queue         M1 解析出的事件
memory.grounded_frame_ids  M2 定位出的关键帧
memory.captions            M3 上下文 caption
memory.grounding           M2 grounding 记录
memory.reasoning_outputs   M3 VQA/OCR 中间答案
memory.plans               M1/M2/M3 解析后的工具调用
memory.raw_plans           LLM 原始工具计划输出
memory.traces              每个阶段的详细工具调用记录
```

### 9.3 日志文件

如果使用：

```bash
nohup python nextqa_demo.py > nextqa_demo_0.log 2>&1 &
```

则日志保存在：

```text
nextqa_demo_0.log
```

日志中会打印：

1. 当前 NExT-QA 样本的问题和选项。
2. M1/M2/M3/Final Prediction 的 prompt。
3. 大模型原始输出。
4. 按论文图示格式打印的 API calls。
5. 每条样本预测结果。
6. 最后总 `accuracy`。

查看准确率：

```bash
grep -n "accuracy" nextqa_demo_0.log
```

从 `jsonl` 文件重新统计准确率：

```bash
python - <<'PY'
import json
path = "outputs/nextqa_val_predictions.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
correct = sum(bool(row.get("correct")) for row in rows)
total = len(rows)
print({"total": total, "correct": correct, "accuracy": correct / total if total else 0})
PY
```

## 10. 日志中的 prompt 和程序输出

本项目为了方便复现论文流程，在日志中打印了模型输入和模型输出。

### 10.1 Prompt 输出

形式为：

```text
===================M1 Event Parsing LLM tool-plan prompt start==================
具体 prompt
===================M1 Event Parsing LLM tool-plan prompt end==================
```

M1、M2、M3 和 Final Prediction 都会打印 prompt。

### 10.2 原始模型输出

形式为：

```text
===================M1 Event Parsing LLM tool-plan model output start==================
{"calls":[...]}
===================M1 Event Parsing LLM tool-plan model output end==================
```

这部分是 Qwen 或 PaliGemma2 的原始返回。

### 10.3 论文风格 API calls 输出

形式为：

```text
===================M1 Event Parsing parsed program API calls start==================
parse_event("none", "two men playing the instrument", "how do the two men play the instrument?")
classify("how")
require_ocr("no")
===================M1 Event Parsing parsed program API calls end==================
```

这部分是把模型输出解析后，以论文图中的 API calls 风格打印出来。

## 11. 百炼 Qwen API 连通性检查

如果 Qwen 一直卡住，不要先跑完整视频任务，先运行：

```bash
python tests/check_dashscope_llm.py --mode both --timeout 25
```

这个脚本会测试：

```text
direct  直连百炼
proxy   使用 configs/LLM_config.yaml 中的 proxy_url 访问百炼
```

结果判断：

```text
direct 成功，proxy 失败：
  把 proxy_url 改为 null。

direct 失败，proxy 成功：
  保留 proxy_url。

direct 和 proxy 都失败：
  检查服务器网络、防火墙、DNS、API Key、模型权限。

GET dashscope 成功，但 chat completions 失败：
  说明访问网页和真正 POST 调模型不是同一回事，需要检查 API Key、模型 code、代理 POST 转发。
```

也可以手动测试：

```bash
curl -I https://dashscope.aliyuncs.com

curl -x http://127.0.0.1:7890 -I https://dashscope.aliyuncs.com
```

真正的模型调用是：

```text
POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
```

所以 `curl -I` 只能证明基础网络可达，不等于模型接口一定可用。

## 12. 常见问题

### 12.1 PaliGemma2 加载很慢

日志中出现：

```text
Loading weights: xx% ...
```

这是在加载 PaliGemma2-10B 的权重。模型较大，第一次加载可能比较慢。

可以用下面命令观察 GPU：

```bash
watch -n 1 nvidia-smi
```

如果只是验证代码流程，可以临时设置：

```python
FORCE_MOCK = True
```

正式实验必须改回：

```python
FORCE_MOCK = False
```

### 12.2 Qwen 一直卡住

优先检查：

1. `api_base` 是否是 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
2. `model` 是否是百炼页面里的模型 code。
3. `api_key` 是否有效。
4. `proxy_url` 是否应该开启。
5. `tests/check_dashscope_llm.py` 是否可以单独调用成功。

## 12. 推荐运行顺序

第一次运行建议按下面顺序：

1. 安装依赖。
2. 确认 `configs/LLM_config.yaml` 中模型路径和 API 配置。
3. 运行百炼连通性测试：

```bash
python tests/check_dashscope_llm.py --mode both --timeout 25
```

4. 使用 mock 跑 1 条样本，验证数据集和流程：

```python
FORCE_MOCK = True
REQUEST_NUM = 1
```

5. 改回真实模型：

```python
FORCE_MOCK = False
REQUEST_NUM = 1
```

6. 确认输出文件、trace 文件和日志正常。
7. 再把 `REQUEST_NUM` 改大，例如 5、20、200。

## 13. 当前项目中最常用的文件

```text
nextqa_demo.py
  NExT-QA 批量实验主入口。

configs/LLM_config.yaml
  所有模型和运行参数配置。

Datasets/loaders.py
  NExT-QA 数据读取逻辑。

framework/morevqa.py
  MoReVQA 总流程。

framework/stages/event_parser.py
  M1 Event Parsing。

framework/stages/grounding.py
  M2 Grounding。

framework/stages/reasoning.py
  M3 Reasoning。

framework/stages/prediction.py
  Final Prediction。

engine/utils.py
  解析 LLM 工具计划、打印 prompt、打印模型输出、打印 API calls。

engine/models/adapters.py
  Qwen、PaliGemma2、OWL-ViT、CLIP 的具体调用方式。

tests/check_dashscope_llm.py
  单独测试百炼 Qwen API。
```

## 14. 重要提醒

1. `FORCE_MOCK = True` 只能验证代码流程，不能用于论文复现实验结果。
2. 正式实验必须使用 `FORCE_MOCK = False`。
3. 真实 API Key 不要写进公开文档，也不要提交到公开仓库。
4. `outputs/nextqa_val_predictions.jsonl` 是总结果文件。
5. `outputs/nextqa_val_traces/` 是每个样本的详细推理过程。
6. 准确率默认打印在日志末尾，不会单独保存成一个 accuracy 文件。
7. 如果需要指定 GPU，在命令前加 `CUDA_VISIBLE_DEVICES="0"`。
