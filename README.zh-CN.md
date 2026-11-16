[English](README.md) | **简体中文**

# H-Mem 与 H-Mem+

本仓库提供 **H-MEM: Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents** 和 **H-Mem+: Source-Augmented Hierarchical Memory for Budgeted Evidence Matching** 的核心实现与可运行示例。

## 运行环境与 API 凭据

| 运行方式 | API 密钥要求 | 运行配置 |
| --- | --- | --- |
| 离线示例 | 无需密钥 | 安装 Python 依赖。示例使用合成对话和轻量模型替身。 |
| 自定义对话数据与本地模型 | 本地服务无需鉴权时免密钥 | 准备对话输入并配置本地模型。H-Mem 支持本地 Ollama。H-Mem+ 需要事件摘要和向量，生成回答时还需接入本地阅读模型。 |
| 自定义对话数据与需鉴权的云端服务 | 需配置相应服务的密钥 | 为记忆提取、向量编码、重排或回答生成所调用的云端服务分别配置 API 密钥。 |

## 安装与示例

使用 Python 3.12。在仓库根目录创建虚拟环境并安装依赖。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r H-Mem/requirements.txt -r 'H-Mem+/requirements.txt'
python -B H-Mem/demo.py
python -B 'H-Mem+/demo.py'
```

也可以进入任一算法目录，安装该目录的 `requirements.txt`，再运行 `python -B demo.py`。自带示例安装依赖后即可在 CPU 上离线运行，无需下载模型或数据集。复现论文的基准结果需要使用论文中的数据集、模型配置和评估协议。

### H-Mem

示例使用哈希编码器和模拟语言模型，将三个属于两个领域的交互记录写入记忆。路由宽度设为 1，检索沿娱乐分支依次经过领域、类别和线索层，找到匹配的事件记忆。输出包括各层节点数、检索路径、参与评分的向量数和模拟回答。

### H-Mem+

示例载入两条已索引的事件摘要和三个会话的原始对话，用词袋向量表示文本、按词项重叠分数重排候选，并以正则表达式计数，在 64 个演示 token 的预算内选取证据。检索从第 2 个会话的原始轮次 `D2:1` 和 `D2:2` 中找回陶艺工作室的关门时间。证据上下文按对话顺序列出选中的来源，每条来源保留一次。

输出包括重排后的候选、选中的来源、因超出预算而跳过的候选、预算用量，以及为阅读模型准备的证据文本。运行 `python -B 'H-Mem+/demo.py' --budget 0` 可查看零预算下的输出。

## 核心实现

| 算法 | 文件 | 功能 |
| --- | --- | --- |
| H-Mem | [extraction.py](H-Mem/hmem/memory/extraction.py)、[store.py](H-Mem/hmem/memory/store.py) | 记忆提取、四层存储和位置索引 |
| H-Mem | [retrieval.py](H-Mem/hmem/memory/retrieval.py)、[update.py](H-Mem/hmem/memory/update.py) | 自顶向下检索与基于反馈的记忆更新 |
| H-Mem+ | [retrieval.py](H-Mem%2B/hmemplus/retrieval.py)、[hybrid_retrieval.py](H-Mem%2B/hmemplus/hybrid_retrieval.py) | 在同一对话内关联事件摘要与来源，并对摘要进行稠密向量检索和词项检索 |
| H-Mem+ | [evidence_budget.py](H-Mem%2B/hmemplus/evidence_budget.py)、[pipeline.py](H-Mem%2B/hmemplus/pipeline.py) | 构建会话内的原始对话窗口、合并候选后重排、按预算打包证据并生成阅读模型的输入 |

## 自定义数据与模型配置

- **H-Mem** 通过 `HMemAgent.ingest(user_text, assistant_text)` 写入一组用户与助手的对话，使用 `retrieve(question)` 检索记忆，或调用 `answer(question)` 生成回答。记忆提取模型和回答模型分别通过 `memory_llm` 和 `llm` 配置。使用 `load_config()` 时，如果云端端点要求鉴权，可分别用 `HMEM_MEMORY_LLM_API_KEY` 和 `HMEM_LLM_API_KEY` 配置对应密钥。编码器和模型端点的配置见 [config.py](H-Mem/hmem/config.py)。
- **H-Mem+** 沿用 H-Mem 的层级结构构建记忆。检索接口接收已存储的事件记录（`pos`、`layer`、`text`、`metadata.source_turns`）、按位置索引组织的事件向量，以及同一对话中按顺序排列的原始轮次。每条事件记录的来源必须在该对话中唯一匹配到一段连续的轮次，输入示例见 [demo.py](H-Mem%2B/demo.py)。`prepare(question, query_vector)` 返回选中的证据及阅读模型的提示词。`answer(question, query_vector)` 通过配置的 `reader(messages)` 回调生成回答。回调接入需鉴权的云端服务时，在对应客户端中配置 API 密钥。
- 论文中的 H-Mem+ 对事件摘要分别进行稠密向量检索和词项检索，也对原始对话窗口进行词项检索，每条检索路径的候选深度为 50。候选合并后用 `cross-encoder/ms-marco-MiniLM-L-6-v2` 重排，只考虑分数最高的 10 个候选，证据预算为 2,048 个 BERT WordPiece。使用论文配置时，事件和问题应使用同一套基于 BERT 的编码器。提供 `rerank(question, documents)` 回调，按输入顺序返回每个候选的分数，再提供 `count_tokens(text)` 回调，使用 BERT 分词器完整计数且不计特殊标记。预算约束作用于格式化后的证据上下文。

## 许可证

原有 MIT 许可证和版权声明见 [LICENSE](LICENSE)。
