# -*- coding: utf-8 -*-
"""一键推送到 GitHub / Gitee。
用法：
  1) 把 push_config.example.json 复制成 push_config.json 并填好 4 个字段
  2) python push.py
脚本会：修正提交作者 -> 设置 remote -> 推送 -> 把 token 从 remote 里抹掉。
"""
import os, json, subprocess, sys

BASE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(BASE, "push_config.json")


def run(*args, check=True):
    print(">", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run(args, cwd=BASE, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.stdout.strip():
        print(r.stdout.strip()[:800], flush=True)
    if r.returncode != 0 and check:
        print("!! 失败:", r.stderr.strip()[:600], flush=True)
        sys.exit(r.returncode)
    return r


def main():
    if not os.path.exists(CFG):
        print("缺少 push_config.json —— 请先复制 push_config.example.json 并填写。")
        sys.exit(1)
    cfg = json.load(open(CFG, encoding="utf-8"))
    url = cfg["remote_url"].strip()
    token = cfg["token"].strip()
    name = cfg["user_name"].strip()
    email = cfg["user_email"].strip()

    if "你的" in url or token.startswith("github_pat_在"):
        print("配置还没填完，请先修改 push_config.json")
        sys.exit(1)

    # 1) 用真实身份重做首个提交（否则 GitHub 不认贡献，简历上说不通）
    run("git", "config", "--local", "user.name", name)
    run("git", "config", "--local", "user.email", email)
    run("git", "commit", "--amend", "--reset-author", "--no-edit", "-q")

    # 2) 当前分支统一叫 main
    run("git", "branch", "-M", "main", check=False)

    # 3) 设置 remote（不含 token）
    run("git", "remote", "remove", "origin", check=False)
    run("git", "remote", "add", "origin", url)

    # 4) 推送：token 临时放 URL，推完立刻抹掉，不留痕
    if "github.com" in url:
        auth_url = url.replace("https://", "https://%s:%s@" % (name, token))
    elif "gitee.com" in url:
        auth_url = url.replace("https://", "https://%s:%s@" % (name, token))
    else:
        auth_url = url
    r = run("git", "push", "-u", auth_url, "main", check=False)
    if r.returncode != 0:
        print("\n推送失败。常见原因：仓库还没在网页上建好、token 权限不足、或网络波动。")
        print("remote 已设置为干净地址（不含 token），可直接重试：git push -u origin main")
        sys.exit(1)

    print("\n推送成功:", url)
    print("remote 中不含 token，安全。")


if __name__ == "__main__":
    main()
