# GitHub 上传指南

> 按顺序执行以下命令即可。

---

## 首次上传

```bash
# 1. 初始化 git 仓库
git init

# 2. 添加所有文件（.gitignore 已配好，会排除 data/、lightgbm/、缓存等）
git add .

# 3. 创建首次提交
git commit -m "初始化量化交易系统

三子系统架构：
- 长期选股（月度多因子排序）
- 盘前推荐（日频隔夜信号融合）
- 日内预测（分钟级实时择时）

8批次开发完成，248个测试通过。"

# 4. 在 GitHub 上创建新仓库
#    打开 https://github.com/new
#    仓库名建议: quant-system 或 a-share-quant
#    ⚠ 不要勾选 "Initialize this repository with a README"
#    创建后会得到一个 URL，类似: https://github.com/你的用户名/quant-system.git

# 5. 关联远程仓库并推送
git remote add origin https://github.com/你的用户名/quant-system.git
git branch -M main
git push -u origin main
```

---

## 后续更新

```bash
git add .
git commit -m "写了什么改动"
git push
```

---

## .gitignore 已排除的内容

| 类别 | 内容 |
|------|------|
| Python 缓存 | `__pycache__/`, `*.pyc`, `.mypy_cache/`, `.ruff_cache/` |
| 虚拟环境 | `.env`, `.venv/` |
| 运行时数据 | `data/` 下全部（checkpoints/models/reports/logs/backups/premarket_recommendations） |
| IDE | `.idea/`, `.vscode/` |
| macOS | `.DS_Store` |
| C++ 编译 | `cpp/build/` |
| 离线安装包 | `lightgbm/`, `lightgbm-*.dist-info/` |

---

## ⚠ 注意：不要提交以下内容到 GitHub

- `.env` 文件中的 token/密码（已通过 .gitignore 排除）
- 从 akshare 拉取的 `.parquet` 行情数据（已在 .gitignore 中按 data/ 目录整体排除）
- lightgbm 手动安装的 dylib 文件
- 任何含真实账户信息、持仓数据的文件
