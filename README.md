# 蜜蜂 Track ID 修正器

用于检查和修正 X-AnyLabeling / LabelMe JSON 中的蜜蜂 Track ID，并支持室外轨迹逐条复查、原视频同步对照、检测框编辑、MOT 导出、自动备份与撤销。

当前版本：`v1.44`

## 获取软件

普通使用者建议从 GitHub Releases 下载 `Bee-TrackID-Corrector-v1.44-Windows.zip`。完整解压后双击 `蜜蜂TrackID修正器_v1.44.exe`，无需安装 Python。

便携包包含：

- Windows 可执行程序和完整运行库；
- 8 段原视频；
- 可重复恢复的演示数据；
- 中文使用说明和演示介绍稿。

## 从源码运行

```powershell
pip install -r requirements.txt
python app.py
```

也可以双击 `run_trackid_corrector.bat`。

如需同步查看原视频，请在程序目录放置 `原视频` 文件夹，或在软件中点击“原视频目录”选择实际路径。

## 数据目录

软件可打开标注员总目录，也可直接打开一个区段目录：

```text
标注员_XX/
├─ A-5-1_区段_01/
│  ├─ A-5-1_frame_000100.jpg
│  ├─ A-5-1_frame_000100.json
│  └─ ...
└─ B-5-1_区段_01/
   └─ ...
```

图片与 JSON 需要同名；JSON 中的蜜蜂矩形框需要带 Track ID。

## 主要模式

### 新 ID 纠正

- 查看新出现的 Track ID 及候选旧 ID；
- `C`：把新 ID 合并到选中的旧 ID；
- `V`：确认为真正的新目标；
- `P`：打开局部运动和原视频对比；
- `G`：打开全局鸟瞰。

### 室外轨迹复查

- 逐条检查名称以 `A-` 开头的室外 Track ID；
- `J / K`：上一条 / 下一条轨迹；
- `Space`：标记当前轨迹通过；
- `M`：标记当前轨迹有问题。

## 局部运动窗口快捷键

| 快捷键 | 功能 |
| --- | --- |
| `A / S` | 上一张 / 下一张抽帧 |
| `D / F` | 上一帧 / 下一帧原视频 |
| `H` | 隐藏 / 恢复检测框 |
| `B` | 隐藏 / 恢复轨迹 |
| `Z` | 显示全部 ID / 当前 ID |
| `X` | 回到目标首次出现位置 |
| `R` | 开启 / 退出画框模式 |
| `Ctrl+Z` | 撤销 |

## 测试

```powershell
python -m pytest -q
```

