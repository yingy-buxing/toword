# toword

本地音频/视频转文字工具，使用 `faster-whisper` + CUDA。它可以直接读取常见音频和视频文件，并输出 `txt`、`srt`、`vtt` 或 `json`。

## 环境准备

建议用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

CUDA 模式还需要可用的 NVIDIA 驱动，以及 `faster-whisper`/`CTranslate2` 能找到的 cuBLAS、cuDNN 运行库。当前 `faster-whisper` 说明里，较新的 `ctranslate2` 主要面向 CUDA 12 + cuDNN 9；如果机器上是旧 CUDA/cuDNN，可能需要固定旧版 `ctranslate2`。

这个工具在 Windows 上会自动尝试复用当前 Python 或基础 Python 里的 `torch\lib` CUDA DLL。如果你的 CUDA DLL 在别处，可以这样指定：

```powershell
python .\transcribe.py "D:\media\demo.mp4" --cuda-dll-dir "E:\Dev\Python\Python312\Lib\site-packages\torch\lib"
```

## 使用

启动图形界面：

```powershell
.\gui.bat
```

界面里选择音频/视频文件、输出目录和参数后，点击“开始转写”即可。

输出纯文本：

```powershell
python .\transcribe.py "D:\media\demo.mp4" --language zh -o .\outputs
```

输出字幕：

```powershell
python .\transcribe.py "D:\media\demo.mp4" --language zh --format srt -o .\outputs
```

更省显存：

```powershell
python .\transcribe.py "D:\media\demo.mp4" --language zh --compute-type int8_float16
```

更快但更吃显存：

```powershell
python .\transcribe.py "D:\media\demo.mp4" --language zh --batch-size 8
```

## 常用参数

- `--model large-v3`: 模型名或本地 CTranslate2 模型目录。显存不够可以用 `medium`、`small`。
- `--device cuda`: 默认使用 CUDA；调试时可改成 `cpu`。
- `--compute-type float16`: CUDA 常用精度；显存紧张可试 `int8_float16`。
- `--language zh`: 指定语言；不填则自动识别。
- `--beam-size 5`: 搜索精度。数值越大可能略准但越慢；一般保持 5。
- `--batch-size 8`: 批处理大小。数值越大通常越快但更吃显存；显存不够就用 0、4 或更小模型。
- `--vad-filter`: 过滤长静音段。
- `--word-timestamps`: 输出词级时间戳，适合后续精细字幕处理，但会更慢。
- `--format txt|srt|vtt|json`: 输出格式。
