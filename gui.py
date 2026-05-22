from __future__ import annotations

import argparse
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from transcribe import (
    prepare_windows_cuda_dlls,
    render_transcript,
    resolve_output_path,
    transcribe,
)


APP_DIR = Path(__file__).resolve().parent


PARAMETER_HELP = {
    "模型": "模型越大越准，但越慢、越吃显存。8GB 显存建议先用 medium；追求速度用 small，追求质量用 large-v3。",
    "语言": "音频里的语言。中文选 zh；不确定就选自动，但指定语言通常更稳。",
    "输出格式": "txt 是纯文本；srt/vtt 是字幕；json 会保留分段时间等结构化信息。",
    "设备": "cuda 使用 NVIDIA 显卡，速度最快；cpu 不用显卡但很慢；auto 让后端自己判断。",
    "精度": "影响速度、显存和准确率。int8_float16 更省显存，适合 8GB 显卡；float16 质量和速度均衡但更吃显存。",
    "任务": "transcribe 是转写原语言；translate 会把语音翻译成英文。",
    "搜索精度 Beam": "模型每一步保留多少个候选结果。数字越大可能略准但更慢；一般 5 就够，想更快可用 1-3。",
    "批处理 Batch": "一次并行处理多少段音频。数字越大通常越快但更吃显存；0 表示不用批处理，8GB 显存建议 0、4 或 8。",
    "过滤静音": "跳过长时间没人说话的部分，通常建议打开，长视频会更快。",
    "词级时间戳": "记录每个词的大致时间。主要给 json 或后续精细字幕用，会更慢。",
    "CUDA DLL 路径": "通常不用填。只有报 cublas64_12.dll/cudnn 之类错误时，再选择包含这些 DLL 的目录。",
}


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event: tk.Event) -> None:
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            self.tip,
            text=self.text,
            padding=(10, 7),
            relief="solid",
            borderwidth=1,
            wraplength=360,
            justify="left",
        )
        label.pack()

    def hide(self, _event: tk.Event) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class TranscribeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("toword - 音视频转文字")
        self.geometry("880x660")
        self.minsize(760, 560)

        self.log_queue: queue.Queue[tuple[str, str | Path | None]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_output: Path | None = None

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(APP_DIR / "outputs"))
        self.model_var = tk.StringVar(value="medium")
        self.language_var = tk.StringVar(value="zh")
        self.format_var = tk.StringVar(value="txt")
        self.device_var = tk.StringVar(value="cuda")
        self.compute_var = tk.StringVar(value="int8_float16")
        self.task_var = tk.StringVar(value="transcribe")
        self.beam_var = tk.IntVar(value=5)
        self.batch_var = tk.IntVar(value=0)
        self.vad_var = tk.BooleanVar(value=True)
        self.word_ts_var = tk.BooleanVar(value=False)
        self.cuda_dir_var = tk.StringVar()
        self.status_var = tk.StringVar(value="准备就绪")

        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        file_frame = ttk.LabelFrame(root, text="文件")
        file_frame.pack(fill="x")
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="输入文件").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(file_frame, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(file_frame, text="选择...", command=self.choose_input).grid(row=0, column=2, padx=8, pady=8)

        ttk.Label(file_frame, text="输出位置").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(file_frame, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(file_frame, text="选择...", command=self.choose_output).grid(row=1, column=2, padx=8, pady=8)

        params = ttk.LabelFrame(root, text="参数")
        params.pack(fill="x", pady=(12, 0))
        for index in range(6):
            params.columnconfigure(index, weight=1)

        self._combo(params, "模型", self.model_var, ("tiny", "base", "small", "medium", "large-v3"), 0, 0)
        self._combo(params, "语言", self.language_var, ("自动", "zh", "en", "ja", "ko", "yue"), 0, 2)
        self._combo(params, "输出格式", self.format_var, ("txt", "srt", "vtt", "json"), 0, 4)

        self._combo(params, "设备", self.device_var, ("cuda", "cpu", "auto"), 1, 0)
        self._combo(params, "精度", self.compute_var, ("int8_float16", "float16", "int8", "float32"), 1, 2)
        self._combo(params, "任务", self.task_var, ("transcribe", "translate"), 1, 4)

        beam_label = ttk.Label(params, text="搜索精度 Beam")
        beam_label.grid(row=2, column=0, sticky="w", padx=8, pady=8)
        ToolTip(beam_label, PARAMETER_HELP["搜索精度 Beam"])
        beam_box = ttk.Spinbox(params, from_=1, to=10, textvariable=self.beam_var, width=10)
        beam_box.grid(
            row=2, column=1, sticky="ew", padx=8, pady=8
        )
        ToolTip(beam_box, PARAMETER_HELP["搜索精度 Beam"])

        batch_label = ttk.Label(params, text="批处理 Batch")
        batch_label.grid(row=2, column=2, sticky="w", padx=8, pady=8)
        ToolTip(batch_label, PARAMETER_HELP["批处理 Batch"])
        batch_box = ttk.Spinbox(params, from_=0, to=32, textvariable=self.batch_var, width=10)
        batch_box.grid(
            row=2, column=3, sticky="ew", padx=8, pady=8
        )
        ToolTip(batch_box, PARAMETER_HELP["批处理 Batch"])

        vad_check = ttk.Checkbutton(params, text="过滤静音", variable=self.vad_var)
        vad_check.grid(
            row=2, column=4, sticky="w", padx=8, pady=8
        )
        ToolTip(vad_check, PARAMETER_HELP["过滤静音"])
        word_ts_check = ttk.Checkbutton(params, text="词级时间戳", variable=self.word_ts_var)
        word_ts_check.grid(
            row=2, column=5, sticky="w", padx=8, pady=8
        )
        ToolTip(word_ts_check, PARAMETER_HELP["词级时间戳"])

        cuda_frame = ttk.LabelFrame(root, text="CUDA DLL 路径（可选）")
        cuda_frame.pack(fill="x", pady=(12, 0))
        cuda_frame.columnconfigure(0, weight=1)
        cuda_entry = ttk.Entry(cuda_frame, textvariable=self.cuda_dir_var)
        cuda_entry.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ToolTip(cuda_entry, PARAMETER_HELP["CUDA DLL 路径"])
        ttk.Button(cuda_frame, text="选择...", command=self.choose_cuda_dir).grid(row=0, column=1, padx=8, pady=8)

        guide = ttk.LabelFrame(root, text="参数怎么选")
        guide.pack(fill="x", pady=(12, 0))
        ttk.Label(
            guide,
            text=(
                "推荐先用：模型 medium，语言 zh，格式 txt，设备 cuda，精度 int8_float16，"
                "搜索精度 Beam=5，批处理 Batch=0。想更快就把模型调小或 Batch 调到 4/8；"
                "显存报错就把 Batch 调回 0，或模型改 small。"
            ),
            wraplength=820,
            justify="left",
        ).pack(fill="x", padx=8, pady=8)

        action_bar = ttk.Frame(root)
        action_bar.pack(fill="x", pady=(12, 0))
        self.start_button = ttk.Button(action_bar, text="开始转写", command=self.start_transcription)
        self.start_button.pack(side="left")
        self.open_button = ttk.Button(action_bar, text="打开输出文件", command=self.open_output, state="disabled")
        self.open_button.pack(side="left", padx=(8, 0))
        ttk.Label(action_bar, textvariable=self.status_var).pack(side="right")

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x", pady=(10, 0))

        log_frame = ttk.LabelFrame(root, text="日志")
        log_frame.pack(fill="both", expand=True, pady=(12, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=14, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

    def _combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
        row: int,
        column: int,
    ) -> None:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=column, sticky="w", padx=8, pady=8)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, width=14)
        combo.grid(
            row=row, column=column + 1, sticky="ew", padx=8, pady=8
        )
        if label in PARAMETER_HELP:
            ToolTip(label_widget, PARAMETER_HELP[label])
            ToolTip(combo, PARAMETER_HELP[label])

    def choose_input(self) -> None:
        path = filedialog.askopenfilename(
            title="选择音频或视频文件",
            filetypes=(
                ("音视频文件", "*.mp3 *.wav *.m4a *.flac *.aac *.ogg *.mp4 *.mkv *.mov *.avi *.webm"),
                ("所有文件", "*.*"),
            ),
        )
        if path:
            self.input_var.set(path)
            if self.output_var.get().strip() == "":
                self.output_var.set(str(Path(path).parent))

    def choose_output(self) -> None:
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_var.set(directory)

    def choose_cuda_dir(self) -> None:
        directory = filedialog.askdirectory(title="选择包含 cublas64_12.dll 的目录")
        if directory:
            self.cuda_dir_var.set(directory)

    def start_transcription(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        input_path = Path(self.input_var.get().strip().strip('"'))
        if not input_path.exists() or not input_path.is_file():
            messagebox.showerror("输入错误", "请选择一个存在的音频或视频文件。")
            return

        output_text = self.output_var.get().strip().strip('"')
        output_path = Path(output_text) if output_text else Path("outputs")
        language = self.language_var.get().strip()
        if language == "自动":
            language = ""

        args = argparse.Namespace(
            input=input_path.resolve(),
            output=output_path,
            format=self.format_var.get(),
            model=self.model_var.get().strip() or "medium",
            device=self.device_var.get().strip() or "cuda",
            compute_type=self.compute_var.get().strip() or "int8_float16",
            cuda_dll_dir=Path(self.cuda_dir_var.get().strip()) if self.cuda_dir_var.get().strip() else None,
            language=language or None,
            task=self.task_var.get().strip() or "transcribe",
            beam_size=int(self.beam_var.get()),
            vad_filter=bool(self.vad_var.get()),
            word_timestamps=bool(self.word_ts_var.get()),
            batch_size=int(self.batch_var.get()),
        )

        self.last_output = resolve_output_path(args.input, args.output, args.format).resolve()
        self.open_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("正在转写...")
        self._clear_log()
        self._append_log(f"输入: {args.input}")
        self._append_log(f"输出: {self.last_output}")
        self._append_log(f"模型: {args.model}, 设备: {args.device}, 精度: {args.compute_type}")

        self.worker = threading.Thread(target=self._run_worker, args=(args,), daemon=True)
        self.worker.start()

    def _run_worker(self, args: argparse.Namespace) -> None:
        try:
            cuda_dir = prepare_windows_cuda_dlls(args.device, args.cuda_dll_dir)
            if cuda_dir is not None:
                self.log_queue.put(("log", f"CUDA DLL: {cuda_dir}"))

            self.log_queue.put(("log", "加载模型中，首次使用会下载模型..."))
            segments, info = transcribe(args, log_callback=lambda line: self.log_queue.put(("log", line)))

            output_path = resolve_output_path(args.input, args.output, args.format).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(render_transcript(segments, info, args.format), encoding="utf-8")
            self.log_queue.put(("done", output_path))
        except Exception as exc:
            self.log_queue.put(("error", str(exc)))

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    self.last_output = Path(str(payload))
                    self._append_log(f"完成: {self.last_output}")
                    self._finish(success=True)
                elif kind == "error":
                    self._append_log(f"失败: {payload}")
                    self._finish(success=False)
                    messagebox.showerror("转写失败", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _finish(self, success: bool) -> None:
        self.progress.stop()
        self.start_button.configure(state="normal")
        self.open_button.configure(state="normal" if success else "disabled")
        self.status_var.set("完成" if success else "失败")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def open_output(self) -> None:
        if not self.last_output or not self.last_output.exists():
            messagebox.showinfo("提示", "还没有可打开的输出文件。")
            return
        os.startfile(self.last_output)


if __name__ == "__main__":
    app = TranscribeApp()
    app.mainloop()
