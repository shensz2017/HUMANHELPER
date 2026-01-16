import asyncio
import base64
import json
import mimetypes
import queue
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, TOP, VERTICAL, W, X, Y, Text, filedialog, messagebox
from tkinter import StringVar, Tk, ttk

import httpx

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
PENDING_DIR = APP_DIR / "Pending processing"
FIRST_SCREEN_DIR = APP_DIR / "First step screening"
SECOND_SCREEN_DIR = APP_DIR / "Second step screening"

DEFAULT_FIRST_PROMPT = (
    "分析这两张图片是否以皮克斯动画的风格转换成功，"
    "人物或动物在数量、特征，服装款式，人物动作，底座数量是否唯一等方面是否符合标准，"
    "是否出现缺胳膊少腿，多手等AI生图普遍错误情况，你只需要用中文回复“正确”和“错误”"
)
DEFAULT_SECOND_PROMPT = (
    "分析两张图片，摆件图的底座是否和白底图的底座在基本的颜色，款式上大体一致，"
    "有略微形状不同可视为正确，你只需要回复“正确”和“错误”"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
GROUP_KEY_REGEX = re.compile(r"([A-Za-z]+\d+)")


@dataclass
class ScreeningTask:
    task_id: str
    images: list[Path]
    prompt: str
    task_type: str
    group_key: str | None = None
    retry_count: int = 0


class AsyncProcessor:
    def __init__(self, api_key_getter, on_result, on_error, on_retry):
        self.api_key_getter = api_key_getter
        self.on_result = on_result
        self.on_error = on_error
        self.on_retry = on_retry
        self.queue: asyncio.Queue[ScreeningTask] = asyncio.Queue()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._worker())
        self.loop.run_forever()

    async def _worker(self):
        semaphore = asyncio.Semaphore(30)
        async with httpx.AsyncClient(timeout=90) as client:
            while True:
                task = await self.queue.get()
                await semaphore.acquire()
                asyncio.create_task(self._handle_task(task, client, semaphore))
                await asyncio.sleep(0.5)

    async def _handle_task(self, task, client, semaphore):
        try:
            result = await self._call_api(task, client)
            self.on_result(task, result)
        except Exception as exc:  # noqa: BLE001
            if task.retry_count < 2:
                task.retry_count += 1
                self.on_retry(task, exc)
                await asyncio.sleep(0.5)
                await self.queue.put(task)
            else:
                self.on_error(task, exc)
        finally:
            semaphore.release()

    async def _call_api(self, task, client):
        api_key = self.api_key_getter()
        if not api_key:
            raise RuntimeError("API key 未设置")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        images_payload = []
        for image_path in task.images:
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type:
                mime_type = "image/jpeg"
            encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            images_payload.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}",
                    },
                }
            )
        content = [{"type": "text", "text": task.prompt}] + images_payload
        payload = {
            "model": "gemini-3-pro",
            "stream": False,
            "messages": [
                {"role": "user", "content": content},
            ],
        }
        response = await client.post(
            "https://grsaiapi.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("API 返回结果为空")
        message = choices[0].get("message") or {}
        return message.get("content", "")

    def submit(self, task: ScreeningTask):
        asyncio.run_coroutine_threadsafe(self.queue.put(task), self.loop)


class ScreeningApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("照片纠错自动筛查")
        self.results_queue: queue.Queue[tuple[ScreeningTask, str] | tuple[ScreeningTask, Exception]] = (
            queue.Queue()
        )
        self.tasks_first: dict[str, ScreeningTask] = {}
        self.tasks_second: dict[str, ScreeningTask] = {}
        self.reference_images: list[Path] = []
        self.api_key_var = StringVar()
        self.first_prompt_var = StringVar(value=DEFAULT_FIRST_PROMPT)
        self.second_prompt_var = StringVar(value=DEFAULT_SECOND_PROMPT)
        self.retry_log: list[str] = []
        self.failed_tasks: dict[str, ScreeningTask] = {}

        self._load_config()
        self._build_ui()

        self.processor_first = AsyncProcessor(
            api_key_getter=lambda: self.api_key_var.get().strip(),
            on_result=self._handle_result,
            on_error=self._handle_error,
            on_retry=self._handle_retry,
        )
        self.processor_second = AsyncProcessor(
            api_key_getter=lambda: self.api_key_var.get().strip(),
            on_result=self._handle_result,
            on_error=self._handle_error,
            on_retry=self._handle_retry,
        )

        self.root.after(200, self._poll_results)

    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=X, padx=10, pady=10)

        ttk.Label(top_frame, text="API Key:").pack(side=LEFT)
        api_entry = ttk.Entry(top_frame, textvariable=self.api_key_var, width=40)
        api_entry.pack(side=LEFT, padx=5)
        ttk.Button(top_frame, text="保存", command=self._save_config).pack(side=LEFT)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.first_tab = ttk.Frame(notebook)
        self.second_tab = ttk.Frame(notebook)

        notebook.add(self.first_tab, text="第一次筛查")
        notebook.add(self.second_tab, text="第二次筛查")

        self._build_first_tab()
        self._build_second_tab()

    def _build_first_tab(self):
        prompt_frame = ttk.Frame(self.first_tab)
        prompt_frame.pack(fill=X, pady=5)

        ttk.Label(prompt_frame, text="批量提示词:").pack(side=LEFT)
        prompt_entry = ttk.Entry(prompt_frame, textvariable=self.first_prompt_var)
        prompt_entry.pack(side=LEFT, fill=X, expand=True, padx=5)

        ttk.Button(prompt_frame, text="填充所有", command=self._fill_first_prompts).pack(side=LEFT)
        ttk.Button(prompt_frame, text="强制覆盖", command=self._overwrite_first_prompts).pack(side=LEFT, padx=5)
        ttk.Button(prompt_frame, text="重新提交选中任务", command=self._resubmit_first_selected).pack(
            side=LEFT, padx=5
        )

        ttk.Button(
            prompt_frame,
            text="选择文件夹",
            command=self._select_first_folder,
        ).pack(side=RIGHT)

        list_frame = ttk.Frame(self.first_tab)
        list_frame.pack(fill=BOTH, expand=True, pady=5)

        columns = ("task", "status")
        self.first_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        self.first_tree.heading("task", text="任务")
        self.first_tree.heading("status", text="状态")
        self.first_tree.column("task", width=400)
        self.first_tree.column("status", width=100, anchor=W)

        scrollbar = ttk.Scrollbar(list_frame, orient=VERTICAL, command=self.first_tree.yview)
        self.first_tree.configure(yscrollcommand=scrollbar.set)
        self.first_tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.first_log = Text(self.first_tab, height=6)
        self.first_log.pack(fill=X, padx=5, pady=5)
        self.first_log.insert("end", "失败日志将在此显示。\n")
        self.first_log.configure(state="disabled")

    def _build_second_tab(self):
        layout = ttk.Frame(self.second_tab)
        layout.pack(fill=BOTH, expand=True)

        left_frame = ttk.Frame(layout)
        left_frame.pack(side=LEFT, fill=Y, padx=5, pady=5)
        ttk.Label(left_frame, text="对照库").pack(anchor=W)
        ttk.Button(left_frame, text="上传对照文件夹", command=self._load_reference_library).pack(
            anchor=W, pady=5
        )
        self.reference_list = ttk.Treeview(left_frame, columns=("file",), show="headings", height=12)
        self.reference_list.heading("file", text="文件")
        self.reference_list.column("file", width=180)
        self.reference_list.pack(fill=Y, expand=True)

        right_frame = ttk.Frame(layout)
        right_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)

        prompt_frame = ttk.Frame(right_frame)
        prompt_frame.pack(fill=X, pady=5)
        ttk.Label(prompt_frame, text="批量提示词:").pack(side=LEFT)
        prompt_entry = ttk.Entry(prompt_frame, textvariable=self.second_prompt_var)
        prompt_entry.pack(side=LEFT, fill=X, expand=True, padx=5)
        ttk.Button(prompt_frame, text="填充所有", command=self._fill_second_prompts).pack(side=LEFT)
        ttk.Button(prompt_frame, text="强制覆盖", command=self._overwrite_second_prompts).pack(
            side=LEFT, padx=5
        )
        ttk.Button(prompt_frame, text="重新提交选中任务", command=self._resubmit_second_selected).pack(
            side=LEFT, padx=5
        )
        ttk.Button(prompt_frame, text="选择文件夹", command=self._select_second_folder).pack(side=RIGHT)

        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill=BOTH, expand=True, pady=5)
        columns = ("task", "status")
        self.second_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        self.second_tree.heading("task", text="任务")
        self.second_tree.heading("status", text="状态")
        self.second_tree.column("task", width=400)
        self.second_tree.column("status", width=100, anchor=W)
        scrollbar = ttk.Scrollbar(list_frame, orient=VERTICAL, command=self.second_tree.yview)
        self.second_tree.configure(yscrollcommand=scrollbar.set)
        self.second_tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.second_log = Text(self.second_tab, height=6)
        self.second_log.pack(fill=X, padx=5, pady=5)
        self.second_log.insert("end", "失败日志将在此显示。\n")
        self.second_log.configure(state="disabled")

    def _load_config(self):
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            self.api_key_var.set(data.get("api_key", ""))

    def _save_config(self):
        CONFIG_PATH.write_text(
            json.dumps({"api_key": self.api_key_var.get().strip()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        messagebox.showinfo("提示", "API Key 已保存")

    def _select_first_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        self._prepare_first_screening(Path(folder))

    def _prepare_first_screening(self, folder: Path):
        PENDING_DIR.mkdir(exist_ok=True)
        self.tasks_first.clear()
        self.failed_tasks.clear()
        for item in self.first_tree.get_children():
            self.first_tree.delete(item)

        groups: dict[str, list[Path]] = {}
        for image_path in self._find_images(folder):
            key = self._extract_group_key(image_path)
            groups.setdefault(key, []).append(image_path)

        for key, images in groups.items():
            if len(images) < 2:
                continue
            images_sorted = sorted(images)[:2]
            dest_folder = PENDING_DIR / key
            dest_folder.mkdir(parents=True, exist_ok=True)
            for img in images_sorted:
                shutil.copy2(img, dest_folder / img.name)
            task = ScreeningTask(
                task_id=key,
                images=[dest_folder / img.name for img in images_sorted],
                prompt=self.first_prompt_var.get(),
                task_type="first",
                group_key=key,
            )
            self.tasks_first[key] = task
            self.first_tree.insert("", "end", iid=key, values=(key, "待处理"))
            self.processor_first.submit(task)

    def _select_second_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        self._prepare_second_screening(Path(folder))

    def _prepare_second_screening(self, folder: Path):
        self.tasks_second.clear()
        self.failed_tasks.clear()
        for item in self.second_tree.get_children():
            self.second_tree.delete(item)

        for image_path in self._find_images(folder):
            task_id = image_path.name
            task_images = [image_path]
            if self.reference_images:
                task_images.extend(self.reference_images)
            task = ScreeningTask(
                task_id=task_id,
                images=task_images,
                prompt=self.second_prompt_var.get(),
                task_type="second",
            )
            self.tasks_second[task_id] = task
            self.second_tree.insert("", "end", iid=task_id, values=(task_id, "待处理"))
            self.processor_second.submit(task)

    def _load_reference_library(self):
        folder = filedialog.askdirectory(title="选择对照库文件夹")
        if not folder:
            return
        self.reference_images = list(self._find_images(Path(folder)))
        for item in self.reference_list.get_children():
            self.reference_list.delete(item)
        for image in self.reference_images:
            self.reference_list.insert("", "end", values=(image.name,))

    def _fill_first_prompts(self):
        for task in self.tasks_first.values():
            if not task.prompt:
                task.prompt = self.first_prompt_var.get()

    def _overwrite_first_prompts(self):
        for task in self.tasks_first.values():
            task.prompt = self.first_prompt_var.get()

    def _fill_second_prompts(self):
        for task in self.tasks_second.values():
            if not task.prompt:
                task.prompt = self.second_prompt_var.get()

    def _overwrite_second_prompts(self):
        for task in self.tasks_second.values():
            task.prompt = self.second_prompt_var.get()

    def _poll_results(self):
        try:
            while True:
                task, result = self.results_queue.get_nowait()
                if isinstance(result, Exception):
                    self._update_task_status(task, f"失败: {result}")
                else:
                    self._update_task_status(task, result)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_results)

    def _handle_result(self, task, result):
        self.results_queue.put((task, result))

    def _handle_error(self, task, exc):
        self.results_queue.put((task, exc))

    def _handle_retry(self, task, exc):
        message = f"[重试 {task.retry_count}/2] {task.task_id}: {exc}"
        self.retry_log.append(message)
        self._append_log(task.task_type, message)

    def _update_task_status(self, task: ScreeningTask, content: str):
        verdict = self._extract_verdict(content)
        if task.task_type == "first":
            tree = self.first_tree
        else:
            tree = self.second_tree

        status_text = "待处理"
        if verdict == "正确":
            status_text = "✅ 正确"
        elif verdict == "错误":
            status_text = "❌ 错误"
        else:
            status_text = "⚠️ 未识别"

        if tree.exists(task.task_id):
            tree.item(task.task_id, values=(task.task_id, status_text))

        if verdict == "" and content.startswith("失败"):
            self.failed_tasks[task.task_id] = task
            self._append_log(task.task_type, f"[失败] {task.task_id}: {content}")

        if verdict == "错误":
            if task.task_type == "first" and task.group_key:
                source_folder = PENDING_DIR / task.group_key
                dest_folder = FIRST_SCREEN_DIR / task.group_key
                dest_folder.parent.mkdir(exist_ok=True)
                shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)
            if task.task_type == "second":
                SECOND_SCREEN_DIR.mkdir(exist_ok=True)
                for image in task.images[:1]:
                    shutil.copy2(image, SECOND_SCREEN_DIR / image.name)

    def _extract_verdict(self, content: str) -> str:
        if not content:
            return ""
        match = re.search(r"(正确|错误)", content)
        return match.group(1) if match else ""

    def _extract_group_key(self, image_path: Path) -> str:
        match = GROUP_KEY_REGEX.search(image_path.stem)
        return match.group(1) if match else image_path.stem

    def _find_images(self, folder: Path):
        for item in folder.iterdir():
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                yield item

    def _append_log(self, task_type: str, message: str):
        if task_type == "first":
            log_widget = self.first_log
        else:
            log_widget = self.second_log
        log_widget.configure(state="normal")
        log_widget.insert("end", f"{message}\n")
        log_widget.see("end")
        log_widget.configure(state="disabled")

    def _resubmit_first_selected(self):
        for item_id in self.first_tree.selection():
            task = self.tasks_first.get(item_id) or self.failed_tasks.get(item_id)
            if not task:
                continue
            task.retry_count = 0
            self.processor_first.submit(task)
            self.first_tree.item(item_id, values=(item_id, "重新提交"))

    def _resubmit_second_selected(self):
        for item_id in self.second_tree.selection():
            task = self.tasks_second.get(item_id) or self.failed_tasks.get(item_id)
            if not task:
                continue
            task.retry_count = 0
            self.processor_second.submit(task)
            self.second_tree.item(item_id, values=(item_id, "重新提交"))


def main():
    root = Tk()
    root.geometry("900x600")
    app = ScreeningApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
