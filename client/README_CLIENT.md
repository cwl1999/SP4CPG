# SP4CPG HGCN + MEDE Windows Client

这是面向 SP4CPG 仓库的 Windows 图形客户端。当前版本已将深度检测核心从 CodeBERT 切换为 HGCN，并保留 MEDE 静态指针分析/静态证据检测与交叉验证流程。

## 1. 启动客户端

```powershell
cd <client目录>
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements-client.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300 --retries 10
python -m sp4cpg_client.app
```

也可以双击 `run_client.bat`。

## 2. HGCN 运行依赖

基础客户端只包含 UI 依赖。若需要真实运行 HGCN 推理或训练，需要在同一 Python 环境中安装图模型依赖：

```powershell
python -m pip install -r requirements-hgcn.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300 --retries 10
```

真实 HGCN 检测需要：

- SP4CPG 仓库根目录；
- `preprocess/hcpg_dataset.pkl`，或在“数据集文件”处直接填写 `.pkl` 的绝对路径；
- 已训练 HGCN `.pt` 模型文件；
- 如需重新生成图工件，可勾选 CPG/HCPG/Embedding 步骤。

如果启用 fallback，客户端在缺少真实 HGCN 输入或模型时会生成明确标记为 `fallback_lexical_not_hgcn` 的结果，仅用于 UI 流程演示，不应作为真实漏洞检测结论。

## 3. MEDE 静态分析

MEDE 部分支持三种方式：

1. 配置 `MEDE_HOME` 与 LLVM bitcode 文件；
2. 填写自定义 MEDE 命令；
3. 直接提供已有静态分析结果 JSON。

如果没有真实 MEDE 环境但启用 fallback，客户端会生成标记为 `fallback_static_not_MEDE` 的静态结果。

## 4. 输出目录

默认输出目录为：

```text
<SP4CPG仓库>/client_outputs/hgcn_workflow/
```

主要输出文件包括：

- `model_predictions.json/.csv`：HGCN 或 fallback 深度检测结果；
- `hgcn_metrics.json`：Accuracy、Precision、Recall、F1 等指标；
- `static_result.json/.csv`：MEDE 或 fallback 静态分析结果；
- `final_report.json/.csv`：交叉验证后的综合结果；
- `hgcn_MEDE_client_report.html`：HTML 报告。

## 5. 总览指标

客户端“总览”页面会读取 `hgcn_metrics.json` 和最终结果表，展示：

- Accuracy；
- Precision；
- Recall；
- F1；
- 高风险告警数量。
