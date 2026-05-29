# Image Compressor

TB 级图片批量压缩工具，支持照片、漫画、CBZ 压缩包。

## 功能

- 支持 JPEG、PNG、WebP 等格式压缩
- 支持 CBZ 漫画压缩包处理
- 断点续传，中断后可恢复
- 多进程并行压缩
- 压缩预设管理
- 丰富的进度显示

## 安装

```bash
# 开发模式安装
pip install -e .

# 或使用 uv
uv pip install -e .
```

## 使用

```bash
# 1. 复制示例配置
cp config.example.toml config.toml

# 2. 扫描目录
image-compressor scan /path/to/images

# 3. 开始压缩
image-compressor compress --preset photo-webp

# 4. 查看进度
image-compressor status

# 5. 安装 shell 补全（可选）
image-compressor --install-completion fish
```

## 命令

| 命令 | 说明 |
|------|------|
| `scan <目录>` | 扫描目录，录入文件到数据库 |
| `compress --preset <预设>` | 开始压缩（断点续传） |
| `status` | 查看进度统计 |
| `compress --dry-run` | 测试模式，只打印不执行 |

## 配置

编辑 `config.toml` 调整压缩参数和预设。

## 架构

详见 [ARCHITECTURE.md](ARCHITECTURE.md)

## 研究计划

详见 [research-plan.md](research-plan.md)
