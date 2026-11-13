# -*- coding: utf-8 -*-
"""下载基座模型到 models/ 目录，默认走国内镜像。
运行：python download_model.py
"""
import os, time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL = os.path.join(BASE, "models", "Qwen2.5-0.5B-Instruct")


def main():
    from huggingface_hub import snapshot_download
    if os.path.exists(os.path.join(LOCAL, "config.json")):
        print("模型已存在，跳过下载:", LOCAL)
        return
    print("下载:", REPO)
    print("镜像:", os.environ["HF_ENDPOINT"])
    t = time.time()
    snapshot_download(REPO, local_dir=LOCAL, max_workers=4)
    print("完成，耗时 %.1f 秒" % (time.time() - t))
    print("路径:", LOCAL)


if __name__ == "__main__":
    main()
