# Image Compressor 架构设计

## 概述

TB 级图片批量压缩工具，支持照片、漫画、CBZ 压缩包。

## 技术栈

| 组件 | 选择 |
|------|------|
| 语言 | Python 3.10+ |
| CLI | typer（支持 tab 补全） |
| 进度显示 | rich（进度条 + 统计面板） |
| 图像处理 | pyvips（主） + Pillow（后备） + CLI 工具（cjxl/avifenc） |
| 配置 | TOML |
| 状态存储 | SQLite |
| 打包 | pyproject.toml + uv/pip |

## 运行环境

### 开发机（批量处理存量）
- CPU: Intel i5-14600KF (14 核 @ 5.30 GHz)
- GPU: NVIDIA RTX 2080
- RAM: 32 GB
- 存储: 本地 SSD + NFS 挂载 NAS
- OS: CachyOS

### NAS（持续运行）
- CPU: Intel Xeon E3-1225 v6 (4 核 @ 3.70 GHz)
- GPU: NVIDIA Tesla P4
- RAM: 16 GB
- 存储: mergerfs 合并 (25 TB)
- OS: NixOS

## 项目结构

```
image-compressor/
├── pyproject.toml
├── README.md
├── config.example.toml
├── src/
│   └── image_compressor/
│       ├── __init__.py
│       ├── cli.py           # CLI 入口 (typer)
│       ├── scanner.py       # 文件发现
│       ├── compressor.py    # 压缩核心 (pyvips + Pillow)
│       ├── cbz_handler.py   # CBZ 解包/打包
│       ├── checkpoint.py    # SQLite 状态管理
│       ├── presets.py       # 压缩预设
│       ├── reporter.py      # rich 进度显示
│       └── config.py        # TOML 配置加载
├── tests/
└── docs/
    └── research-plan.md
```

## CLI 命令

```bash
# 扫描目录，录入文件到数据库
image-compressor scan <目录路径> [--recursive]

# 压缩（断点续传）
image-compressor compress --preset <预设名> [--workers N] [--dry-run]

# 查看进度
image-compressor status [--detailed]

# 安装 shell 补全
image-compressor --install-completion fish|bash|zsh
```

## 压缩预设

具体参数待研究测试确定。预设结构：

```toml
[presets.<name>]
description = "..."
strategy = "lossy|lossless|auto"
format = "webp|jpeg-xl|avif|keep"
quality = 85  # 有损时的质量
process_cbz = true|false
```

## 数据库设计

```sql
-- 文件处理状态
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    source_type TEXT NOT NULL,        -- 'file' | 'cbz_entry'
    parent_cbz TEXT,                  -- CBZ 内部文件的父 CBZ 路径
    original_size INTEGER,
    compressed_size INTEGER,
    status TEXT DEFAULT 'pending',    -- pending | processing | completed | failed | skipped
    error_message TEXT,
    processing_time REAL,
    preset TEXT,                      -- 使用的预设
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_files_status ON files(status);
CREATE INDEX idx_files_parent_cbz ON files(parent_cbz);
```

## 压缩流程

```
1. scan: 递归扫描目录 → 插入 files 表 (status=pending)
2. compress:
   a. 查询 status=pending 的文件
   b. 对每个文件：
      - 更新 status=processing
      - 根据预设选择压缩策略
      - 压缩到临时文件
      - 移动原文件到 .original/
      - 移动压缩文件到原位置
      - 更新 status=completed + 压缩结果
   c. CBZ 文件：
      - 解包到临时目录
      - 压缩内部图片
      - 重新打包 (ZIP_STORED)
      - 替换原 CBZ
```

## 错误处理

| 错误类型 | 处理 |
|---------|------|
| 图片解码失败 | 记录 status=failed，跳过 |
| 压缩工具报错 | 记录错误，跳过 |
| CBZ 解包失败 | 记录，跳过整个 CBZ |
| 磁盘空间不足 | 立即停止 |
| 中断 (Ctrl+C) | 优雅退出，保存进度 |

## 后续阶段

- [ ] systemd 服务模式（daemon）
- [ ] FastAPI WebUI（进度监控 + 控制）
- [ ] inotify 文件监控（新增文件实时处理）
- [ ] GPU 加速编码（NVENC / Tesla P4）
