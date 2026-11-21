# 浮声 · 翻唱工作台

把一首歌里的人声换成目标音色（伴奏保留），全程在本机跑，不上传任何服务器。
界面分 **翻唱** 和 **训练模型** 两页：翻唱页按 4 步走完就得到成品；训练页可以用自己的干声做出新的音色模型。

- 自带音色模型：**FeiYuVoice**（RVC v2 · 40 kHz · 带 FAISS 检索索引）
- 人声分离：**BS-RoFormer**（模型分离）／**官方伴奏相减**（需要同版本伴奏）
- 换音色：RVC 推理（RMVPE 提取音高，检索索引 0.75 混合）
- 混音母带：人声/伴奏按 50 ms RMS 包络做电平匹配 → alimiter → loudnorm −11 LUFS，输出 320k MP3 + 无损 WAV

---

## 一、安装

需要 **Windows + NVIDIA 显卡（CUDA 12.1）+ Python 3.12**。

1. 安装 [Python 3.12](https://www.python.org/downloads/)（安装时勾选 *Add python.exe to PATH*）。
2. 双击 **`setup.bat`**，它会依次：
   - 建好 `.venv` 虚拟环境；
   - 安装 PyTorch 2.4.0（CUDA 12.1）；
   - 安装 `requirements.txt` 里的依赖（FastAPI、librosa、faiss…）；
   - 安装 BS-RoFormer 后端 `pymss`（用 `--no-deps`，因为它的元数据要求 torch≥2.7.1，但实测 2.4.0 可用，不装依赖就不会把 torch 顶掉）。
3. 安装完成后双击 **`启动.bat`**，浏览器会自动打开 `http://127.0.0.1:7860`。

手工安装等价于：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps pymss==2.1.12 pymss-core==0.1.14
.\.venv\Scripts\python.exe webapp\app.py
```

> 换端口：`set PORT=7861` 后再启动。端口被占用时再点 `启动.bat` 只会打开浏览器，不会重复起服务。

### 目录结构

```
浮声/
├─ 启动.bat                 启动服务并打开浏览器
├─ setup.bat                一次性安装依赖
├─ requirements.txt         运行依赖
├─ webapp/
│  ├─ app.py                FastAPI 后端（翻唱 4 步 + 训练）
│  ├─ index.html            前端（单文件，无构建步骤）
│  └─ models.json           模型登记表（默认模型 = FeiYuVoice）
├─ work/
│  └─ subtract_vocal.py     官方伴奏相减（对齐 + 逐频带增益 + 相减）
└─ tools/
   ├─ bin/ffmpeg.exe        音频转码/母带
   ├─ sep_roformer.py       BS-RoFormer 分离入口（pymss 后端）
   ├─ pymss_weights/        BS-RoFormer 权重（ckpt + yaml）
   └─ rvc_src/Retrieval-based-Voice-Conversion-WebUI-main/
      ├─ infer/             RVC 推理（rmvpe / hubert / faiss）
      ├─ train/             训练流程（切片 → f0 → 特征 → 训练 → 索引 → 导出）
      ├─ configs/ i18n/ tools/
      └─ assets/
         ├─ weights/FeiYuVoice.pth      音色权重（RVC v2 单文件整包）
         ├─ indices/FeiYuVoice.index    检索索引
         ├─ hubert_base/                ContentVec 特征提取
         ├─ rmvpe/                      RMVPE 音高提取
         └─ pretrained_v2/              训练用的预训练底座
```

---

## 二、翻唱页（4 步）

| 步骤 | 做什么 | 产物 |
|---|---|---|
| 1 获取源音频 | 上传本地音频，或粘 B 站 BV/链接/直链 | 44.1 kHz WAV + 原始文件 |
| 2 分离人声与伴奏 | BS-RoFormer 模型分离／官方伴奏相减；源是纯人声可直接跳过 | 人声、伴奏各一条 |
| 3 换音色 | 选模型（FeiYuVoice）、变调 −24~+24 | 换声后的人声 |
| 4 混合与母带 | 电平匹配 + 限幅 + 响度母带 | 成品 MP3 320k + 成品 WAV |

- 每一步的产物都带播放条和「另存为」，可以先试听再决定要不要继续／重跑。
- 没完成前置步骤时**只禁用执行按钮**，参数可以提前选好。
- 重跑某一步会作废它后面的产物，避免界面指向过期文件。
- 任务目录：`webapp/sessions/<10 位 ID>/`，可重命名、删除、打开文件夹；`session.json` 记录每一步的状态和参数。

### 分离方式：BS-RoFormer / 官方伴奏相减

- **BS-RoFormer 模型分离**（默认）：直接对整首歌做分离。
- **官方伴奏相减**：需要原曲和官方伴奏是**同一版本**，否则比模型分离更脏（页面会用低频相干度提示）。
  选中它之后，卡片里会展开伴奏输入区：可以**上传本地音频**，也可以贴 **BV 号 / 链接** 让它自己下；拿到的伴奏和原始文件都能「另存为」留档。
  相减的流程（`work/subtract_vocal.py`）：
  1. **对齐**：8 kHz 包络粗搜 → 全采样率互相关 + 抛物线插值得到亚样本延时 → FFT 相位斜坡做分数延迟；
  2. **自动匹配音量**：2 秒一块、逐频带估计复数增益，块间按“是否真的在变”决定跟不跟；
  3. **相减**：人声 = 原曲 − g(f,t)·伴奏。

  分离卡片里会显示 `自动对齐 +0.0 ms · 音量补偿 +6.2 dB` 这样的诊断。
- **跳过分离**：勾上「源音频已是纯人声，跳过分离」**立刻生效**，不用再点「开始分离」，下一步直接解锁；取消勾选会作废当前的分离结果，回到可以重新分离的状态。

### 两条常用路径

1. **整首换声**：上传原曲 → 分离 → 换音色 → 混合。
2. **自己唱、只换音色**：上传清唱 → 勾「源音频已是纯人声，跳过分离」→ 换音色 → 混合。
   如果另外放了官方伴奏，第 4 步直接叠加，**不会自动对齐**，两份文件起点要一致。

### 变调怎么填

| 场景 | 建议 |
|---|---|
| 男声 → 女声音色 | +10 ~ +12 半音（以耳朵为准） |
| 只改音色、不改音高 | 0 |

> RVC 升调会连共振峰一起抬，偏高就退一档。

---

## 三、训练模型页

上传目标音色的**干声**（无伴奏、无混响、无人声以外的噪声），填模型名、选轮数即可。

| 轮数 | 约耗时（2 分钟素材 / batch 4） |
|---|---|
| 60 | 10 分钟 |
| 120（推荐） | 20 分钟 |
| 200 | 35 分钟 |
| 300 | 50 分钟 |

流程：切片（3.7 s）→ RMVPE 提取音高 → ContentVec 特征 → 从 `f0G40k/f0D40k` 起训 → 建 FAISS 索引 → 导出到 `assets/weights/<名字>.pth` → 自动登记到 `webapp/models.json`。

- 训练与推理共用显卡，同一时刻只跑一个 GPU 任务，另一个排队。
- 训练中可以「取消训练」；中途刷新页面会接着显示进度。
- 勾选「训练完自动关机」会在训练**成功**后执行 `shutdown /s /f /t 120`（有 120 秒缓冲，页面可取消）。

---

## 四、HTTP 接口

每一步都是异步的：POST 立刻返回，前端轮询 `GET /api/sessions/{sid}` 的 `job` 字段。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 界面 |
| POST／GET | `/api/sessions` | 新建任务／最近任务 |
| GET | `/api/sessions/{sid}` | 任务状态 + 各步产物 |
| POST | `/api/sessions/{sid}/rename` | `name`：重命名 |
| POST | `/api/sessions/{sid}/reveal` | 在资源管理器打开任务目录 |
| DELETE | `/api/sessions/{sid}` | 删除任务整个目录（处理中返回 409） |
| POST | `/api/sessions/{sid}/source` | `file` 或 `url` |
| POST | `/api/sessions/{sid}/inst` | `file` 或 `url`：官方伴奏 |
| POST | `/api/sessions/{sid}/inst/clear` | 清除官方伴奏 |
| POST | `/api/sessions/{sid}/separate` | `method=roformer/inst`、`skip=true/false` |
| POST | `/api/sessions/{sid}/sep/clear` | 取消“跳过分离”，作废分离结果 |
| POST | `/api/sessions/{sid}/convert` | `model`、`pitch` |
| POST | `/api/sessions/{sid}/mix` | 混音母带 |
| GET | `/files/{sid}/{path}` | 取产物 |
| GET | `/api/models` | 模型列表 + 默认模型 |
| POST | `/api/models/default` | `name`：设为默认 |
| POST | `/api/train` | 训练（multipart：`name`/`epochs`/`batch`/`files`） |
| GET | `/api/jobs/{id}`、`/api/jobs` | 训练进度 |
| POST | `/api/cancel/{id}` | 取消训练 |
| POST | `/api/shutdown/cancel` | 取消关机 |

---

## 五、模型说明

`FeiYuVoice.pth` 是 **RVC v2 单文件整包**（顶层键 `weight/config/version/sr/f0/info`，约 27.5 M 参数），推理只需要它 + `hubert_base` + `rmvpe`，不需要额外的 GPT `.ckpt`、参考音频或 3~4 GB 底模。
`FeiYuVoice.index` 是 FAISS 检索索引，`index_rate=0.75`；没有索引也能跑，只是音色细节少一点。

换成别的 RVC 模型：把 `.pth` 放进 `assets/weights/`，在 `webapp/models.json` 里补一条（`path`、`index`、`sr`、`f0`、`default`），刷新页面即可。`assets/weights/` 里未登记的 `.pth` 会被自动发现。

---

## 六、常见问题

| 现象 | 处理 |
|---|---|
| 双击 `启动.bat` 提示缺少 `.venv` | 先跑 `setup.bat` |
| 显存不足 | 关掉占显存的程序；服务端已设 `RVC_CUDA_GRAPH=0` |
| 换声后很沙 | 源分离质量差（低码率／强混响）；换无损源，或先降噪 |
| 官方伴奏相减有残渣 | 两份不是同一版本／没对齐；改用 BS-RoFormer 分离 |
| 训练报「素材太短」 | 少于 5 段（约 20 秒）训不了，多给一点干声 |
| 声音偏闷 | 训练素材本身可能带限（高频不足），换更好的素材重训 |
| 想删任务 | 当前任务点「删除任务」，或删 `webapp/sessions/<任务ID>/` |
| 想删模型 | 删 `assets/weights/<名>.pth` 与 `models.json` 里对应条目 |

> 纯人声 + 官方伴奏的混合**不自动对齐**（按文件起点叠加），需要起点一致。

---

## 七、第三方组件与许可

| 组件 | 用途 | 许可 |
|---|---|---|
| [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | RVC 推理与训练代码（`tools/rvc_src`） | MIT |
| [ContentVec / hubert_base](https://huggingface.co/lj1995/VoiceConversionWebUI) | 语音特征 | 见上游 |
| RMVPE | 音高提取 | 见上游 |
| [BS-RoFormer](https://github.com/ZFTurbo/Music-Source-Separation-Training) / [pymss](https://github.com/pymss-project/pymss) | 人声/伴奏分离 | MIT / 权重见上游 |
| [FFmpeg](https://ffmpeg.org/)（`tools/bin/ffmpeg.exe`，gyan.dev essentials build） | 转码、母带 | GPL v3（二进制随附，源码见 ffmpeg.org） |
| [FastAPI](https://fastapi.tiangolo.com/) / [librosa](https://librosa.org/) / [soundfile](https://python-soundfile.readthedocs.io/) | 后端与音频处理 | MIT / ISC |

本项目（浮声）代码（`webapp/`、`work/`、`tools/sep_roformer.py`、脚本）采用 MIT 许可，详见 `LICENSE`。
模型权重 `FeiYuVoice.pth` 由本项目作者训练并随附，仅供个人学习与非商业用途；**请勿用于伪造他人声音或任何违法用途**。
