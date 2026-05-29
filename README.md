# MoReVQA 开放模型替代复现项目

本项目依据 CVPR 2024 论文 **MoReVQA: Exploring Modular Reasoning Models for Video Question Answering** 实现 MoReVQA 主流程，并按照 [CLOVA-tool](https://github.com/clova-tool/CLOVA-tool) 的组织方式整理代码。

论文中的 `PaLI-3 (5B)` 和 `PaLM-2` 无法由普通用户直接获得。本项目保持先前确定的开放模型替换方案：

| 论文组件 | MoReVQA 中的职责 | 本项目配置 |
| --- | --- | --- |
| PaLI-3 (5B) | 帧描述、视觉问答、OCR | `google/paligemma2-10b-mix-448` |
| PaLM-2 | M1/M2/M3 计划生成与最终预测 | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| OWL-ViT | 实体与区域定位 | `google/owlvit-base-patch32` |
| CLIP RN50 | 图文相似度匹配 | OpenAI CLIP `RN50` |

由于核心闭源模型已经替换，本项目能够复现 MoReVQA 的模块化调用流程和实验组织方式，但不能声称严格复现论文原始数值。

## 当前范围

当前代码只提供 **MoReVQA pipeline**：

1. `M1` Event Parsing：解析问题中的事件、时序关系、问题类型和 OCR 需求。
2. `M2` Grounding：使用 OWL-ViT、CLIP 和视觉问答能力定位相关帧。
3. `M3` Reasoning：对 grounded frames 与全局 context frames 执行视觉推理。
4. Final Prediction：由 Qwen3 基于外部记忆输出答案。

当前支持单个视频问题运行并保存 JSON trace。数据集批量读取器、全量评测器及论文扩展任务指标仍未实现。

## 与 CLOVA 的结构对应

CLOVA 的根目录 demo 调用 `framework` 中的任务主类，主类使用 `engine.utils` 中的程序生成器与解释器连接 `prompts` 和 `tools`。本项目采用相同层次，同时保留 MoReVQA 论文要求的 M1/M2/M3 三阶段顺序。

```text
MoReVQA_project/
  configs/
    LLM_config.yaml           主模型与推理配置
  Datasets/
    README.md                 数据集接入说明；后续 loader/evaluator 放在此处
  data/                       实际视频与标注数据
  model/                      本地模型目录，每个模型一个子文件夹
  engine/
    config.py                 YAML 配置读取
    utils.py                  ProgramGenerator / ProgramInterpreter
    memory.py                 外部记忆与 trace
    schemas.py                共享数据结构
    video.py                  视频抽帧工具
    models/
      adapters.py             Qwen、PaliGemma、OWL-ViT、CLIP 调用适配器
      registry.py             按配置构造模型实例
  framework/
    morevqa.py                MoReVQA 主类，提供 inference(...)
    stages/
      event_parser.py         M1
      grounding.py            M2
      reasoning.py            M3
      prediction.py           最终预测
  prompts/
    prompt_engineering.py     三阶段与最终预测提示词
  tools/
    event.py                  M1 可执行 API
    grounding.py              M2 可执行 API
    reasoning.py              M3 可执行 API
  videoqa_demo.py             与 CLOVA demo 对应的可运行入口
  main.py                     兼容入口，转发到 videoqa_demo.py
  morevqa/
    cli.py                    命令行入口
  requirements.txt
```

真实推理的调用链是：

```text
videoqa_demo.py
  -> framework.morevqa.MoReVQA.inference(...)
     -> framework.stages (M1 -> M2 -> M3 -> Prediction)
        -> engine.utils.ProgramGenerator / ProgramInterpreter
           -> prompts/ 与 tools/
              -> engine.models 中配置的开放模型
```

## 论文设置与配置

主配置文件为 `configs/LLM_config.yaml`。

| 设置项 | 论文设置 | 本项目设置 |
| --- | --- | --- |
| MoReVQA 上下文帧 | 均匀采样 `16` 帧 | `video.context_frames: 16` |
| EgoSchema 上下文帧 | `30` 帧 | 运行该数据集前改为 `30` |
| LLM 温度 | `0` | `llm.temperature: 0.0` |
| 定位器 | OWL-ViT | `google/owlvit-base-patch32` |
| 图文匹配 | CLIP RN50，阈值 `0.7` | `RN50`，`threshold: 0.7` |

正式运行时请保持：

```yaml
runtime:
  mock_on_missing: false
```

`FORCE_MOCK = True` 只用于检查代码连通性，不能用于报告实验结果。

## 一、创建运行环境

在项目根目录打开 PyCharm Terminal 或 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

可选：安装本项目命令行入口。

```powershell
pip install -e . --no-deps
```

主项目环境负责视频解码、PaliGemma 2、OWL-ViT 和 CLIP。Qwen3 推荐由 Linux/CUDA 环境中的 vLLM 独立提供 HTTP 服务，因此 `vllm` 不写入 Windows 主项目的 `requirements.txt`。

## 二、准备 Hugging Face 权限

### 2.1 接受 PaliGemma 许可

1. 登录 Hugging Face。
2. 打开 [google/paligemma2-10b-mix-448](https://huggingface.co/google/paligemma2-10b-mix-448)。
3. 按页面提示接受 Gemma 使用许可。
4. 在 Hugging Face 设置中创建具有读取权限的 token。

### 2.2 在当前环境登录

如果输入 `hf auth login` 提示找不到命令，可统一使用 Python 模块方式：

```powershell
python -m pip install -U "huggingface_hub[cli]"
python -m huggingface_hub.commands.huggingface_cli login
```

根据提示粘贴 token。验证登录状态：

```powershell
python -m huggingface_hub.commands.huggingface_cli whoami
```

## 三、下载模型到 `model/`

模型目录保持为你已经创建的 `model/`，每个模型单独存放。

### 3.1 PaliGemma 2

```powershell
python -m huggingface_hub.commands.huggingface_cli download google/paligemma2-10b-mix-448 --local-dir model/paligemma2-10b-mix-448
```

将 `configs/LLM_config.yaml` 中 `vision_language.captioner.model_path` 和 `vision_language.vqa.model_path` 均改为：

```yaml
model_path: model/paligemma2-10b-mix-448
```

PaliGemma 在项目中复用同一实例：

| 功能 | 提示格式 |
| --- | --- |
| 帧描述 | `caption en` |
| 视觉问答 | `answer en {question}` |
| OCR | `ocr` |

### 3.2 OWL-ViT

```powershell
python -m huggingface_hub.commands.huggingface_cli download google/owlvit-base-patch32 --local-dir model/owlvit-base-patch32
```

将配置改为：

```yaml
grounding:
  detector:
    model_path: model/owlvit-base-patch32
```

### 3.3 CLIP RN50

`requirements.txt` 会安装 OpenAI CLIP 代码。第一次真实运行 `clip.load("RN50")` 时会下载 RN50 权重；配置无需更换模型名称：

```yaml
grounding:
  image_text_scorer:
    provider: openai_clip_rn50
    model_name: RN50
    threshold: 0.7
```

### 3.4 Qwen3

Qwen3 由 vLLM 提供 OpenAI-compatible 服务。在 Linux/CUDA 环境中执行：

```bash
python -m venv qwen_server_env
source qwen_server_env/bin/activate
python -m pip install --upgrade pip
pip install -U vllm
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 127.0.0.1 --port 8000
```

多 GPU 时可加入例如：

```bash
--tensor-parallel-size 2
```

主项目的连接配置为：

```yaml
llm:
  provider: openai_compatible
  api_base: http://127.0.0.1:8000/v1
  api_key: EMPTY
  model: Qwen/Qwen3-30B-A3B-Instruct-2507
  temperature: 0.0
```

若服务运行在另一台机器，请将 `api_base` 改为该机器可访问的地址。

## 四、准备数据集目录

实际数据存放在 `data/`，`Datasets/` 用于将来加入 Python loader 与 evaluator，不需要将视频搬到 `Datasets/`。

```text
data/
  nextqa/{videos,annotations}
  ivqa/{videos,annotations}
  egoschema/{videos,annotations}
  activitynet_qa/{videos,annotations}
  next_gqa/{videos,annotations}
  activitynet_paragraphs/{videos,annotations}
```

| 数据集 | 用途 | 获取入口 | 当前可运行范围 |
| --- | --- | --- | --- |
| NExT-QA | 多项选择 VideoQA | [doc-doc/NExT-QA](https://github.com/doc-doc/NExT-QA) | 单视频问题 |
| iVQA | 开放式 VideoQA | [antoyang/just-ask](https://github.com/antoyang/just-ask)；视频来自 [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) | 单视频问题 |
| EgoSchema | 长视频多项选择 | [EgoSchema](https://egoschema.github.io/) | 单视频问题；需使用 30 帧 |
| ActivityNet-QA | 开放式 VideoQA | [ActivityNet-QA](https://github.com/MILVLG/ActivityNet-Question-Answering) | 单视频问题 |
| NExT-GQA | Grounded VideoQA | [doc-doc/NExT-GQA](https://github.com/doc-doc/NExT-GQA) | 尚无官方指标实现 |
| ActivityNet-Paragraphs | 段落描述 | [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) | 尚无生成与 CIDEr 评测 |

其中 EgoSchema 运行前，需要在 `configs/LLM_config.yaml` 修改：

```yaml
video:
  context_frames: 30
```

其他 VideoQA 数据集默认使用：

```yaml
video:
  sample_fps: 1.0
  context_frames: 16
```

## 五、在 PyCharm 中直接运行

打开根目录 `videoqa_demo.py`，修改文件顶部的变量：

```python
VIDEO_PATH = PROJECT_ROOT / "data" / "nextqa" / "videos" / "VIDEO_ID.mp4"
QUESTION = "QUESTION_TEXT"
OPTIONS = ["ANSWER_0", "ANSWER_1", "ANSWER_2", "ANSWER_3", "ANSWER_4"]
TRACE_PATH = PROJECT_ROOT / "outputs" / "trace.json"
FORCE_MOCK = False
```

开放式问答数据集把 `OPTIONS` 改为：

```python
OPTIONS = None
```

操作步骤：

1. 在 PyCharm 中选择 `.venv` 作为项目解释器。
2. 确保 PaliGemma 2 与 OWL-ViT 已可读取，CLIP 可自动下载。
3. 确保 Qwen3 vLLM 服务已启动。
4. 右键 `videoqa_demo.py`，点击 Run。
5. 从 `TRACE_PATH` 查看 M1、M2、M3 和最终预测的完整 JSON 记录。

根目录 `main.py` 仍能运行，但仅作为兼容入口，实验参数应在 `videoqa_demo.py` 中配置。

## 六、命令行运行单样例

### 6.1 Mock 流程检查

需提供一个有效视频文件，但不会加载真实模型：

```powershell
python -m morevqa.cli --mock --video examples/demo_video.mp4 --question "What is happening in the video?" --options "option one" "option two" --trace outputs/mock_trace.json
```

### 6.2 NExT-QA

```powershell
python -m morevqa.cli --config configs/LLM_config.yaml --video data/nextqa/videos/VIDEO_ID.mp4 --question "QUESTION_TEXT" --options "ANSWER_0" "ANSWER_1" "ANSWER_2" "ANSWER_3" "ANSWER_4" --trace outputs/nextqa_morevqa_trace.json
```

### 6.3 iVQA

```powershell
python -m morevqa.cli --config configs/LLM_config.yaml --video data/ivqa/videos/VIDEO_ID.mp4 --question "QUESTION_TEXT" --trace outputs/ivqa_morevqa_trace.json
```

### 6.4 EgoSchema

先将 `video.context_frames` 改成 `30`，再执行：

```powershell
python -m morevqa.cli --config configs/LLM_config.yaml --video data/egoschema/videos/VIDEO_ID.mp4 --question "QUESTION_TEXT" --options "ANSWER_0" "ANSWER_1" "ANSWER_2" "ANSWER_3" "ANSWER_4" --trace outputs/egoschema_morevqa_trace.json
```

### 6.5 ActivityNet-QA

```powershell
python -m morevqa.cli --config configs/LLM_config.yaml --video data/activitynet_qa/videos/VIDEO_ID.mp4 --question "QUESTION_TEXT" --trace outputs/activitynet_qa_morevqa_trace.json
```

## 七、输出内容

设置 `--trace` 或 `TRACE_PATH` 后，输出 JSON 包含：

| 字段 | 内容 |
| --- | --- |
| `question` / `options` | 输入问题与候选答案 |
| `plans.event_parsing` | M1 的工具程序 |
| `plans.grounding` | M2 的工具程序 |
| `plans.reasoning` | M3 的工具程序 |
| `grounding` | 相关帧、检测和相似度结果 |
| `captions` | 上下文帧描述 |
| `reasoning_outputs` | 视觉子问题答案 |
| `prediction` | 最终答案 |

## 八、测试与问题排查

运行测试：

```powershell
python -m pytest -q
```

| 情况 | 处理方式 |
| --- | --- |
| PaliGemma 无权访问 | 在模型页面接受许可，再重新登录 Hugging Face |
| `hf` 命令不可用 | 使用本文给出的 `python -m huggingface_hub.commands.huggingface_cli ...` 形式 |
| Qwen 请求连接失败 | 检查 vLLM 是否运行，以及 `llm.api_base` 地址 |
| GPU 显存不足 | 单独部署 Qwen3、调整 vLLM 并行参数，或先用 mock 验证代码 |
| 输出来自 mock | 检查 `FORCE_MOCK = False` 和 `mock_on_missing: false` |

## 资源链接

| 类型 | 链接 |
| --- | --- |
| 论文 | [MoReVQA CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Min_MoReVQA_Exploring_Modular_Reasoning_Models_for_Video_Question_Answering_CVPR_2024_paper.html) |
| 结构参考 | [clova-tool/CLOVA-tool](https://github.com/clova-tool/CLOVA-tool) |
| VLM | [google/paligemma2-10b-mix-448](https://huggingface.co/google/paligemma2-10b-mix-448) |
| LLM | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) |
| LLM 部署 | [vLLM OpenAI-compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) |
| Grounding | [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) |
| Matching | [openai/CLIP](https://github.com/openai/CLIP) |

## NExT-QA 批量运行

本项目现在提供了和 CLOVA 任务 demo 类似的 NExT-QA 数据读取方式。目录保持为：

```text
data/nextqa/
  annotations/
    map_vid_vidorID.json
    train.csv
    val.csv
    test.csv
  videos/
    0000/
      *.mp4
    0001/
      *.mp4
```

`Datasets.loaders.NextQADataset` 会读取 `annotations/{split}.csv`，再通过 `map_vid_vidorID.json` 找到 `videos/<子目录>/<video_id>.mp4`。你可以用 `data_num` 控制一次传给 MoReVQA pipeline 的请求数量。

### PyCharm 直接运行

打开根目录的 `nextqa_demo.py`，修改顶部变量：

```python
SPLIT = "val"
REQUEST_NUM = 5
START_INDEX = 0
INTERVAL = 1
SKIP_MISSING_VIDEOS = True
FORCE_MOCK = False
```

然后右键 `nextqa_demo.py`，点击 Run。结果会写入：

```text
outputs/nextqa_val_predictions.jsonl
outputs/nextqa_val_traces/
```

### 命令行运行

只跑验证集前 10 条：

```powershell
python -m morevqa.cli --dataset nextqa --config configs/LLM_config.yaml --data-root data/nextqa --split val --num-samples 10 --output outputs/nextqa_val_predictions.jsonl --trace-dir outputs/nextqa_val_traces
```

从第 100 条开始，每隔 5 条取一个样本，总共跑 20 个请求：

```powershell
python -m morevqa.cli --dataset nextqa --config configs/LLM_config.yaml --data-root data/nextqa --split val --start-index 100 --interval 5 --num-samples 20 --output outputs/nextqa_val_sparse_predictions.jsonl
```

如果你要严格检查视频是否完整，可以加上 `--keep-missing-videos`，这样找不到视频时会直接报错，而不是跳过该样本。
