# Gakumas Assistant 使用指北

## 安装前的碎碎念
### 是否支持汉化版
当前版本暂未支持

### 是否支持DMM版本
当前版本主要为DMM版本开发，手机适配上可能会出现问题

### 是否支持手机？只能电脑用吗？
暂时无移植到手机上的计划，实时图像推理对性能的需求会比较高

### 是否支持非标分辨率
按理来说是支持的，在程序设计时就采用的是全目标识别的架构

## 下载和安装
### 系统要求
在开始安装 Gakumas Assistant 之前：  
如果你打算在安卓模拟器上使用 Gakumas Assistant，先检查模拟器是否满足这些设置要求：  
- 系统版本：Android 10+  
- 支持Google Play
- 能正常使用ADB
- 已开启游戏加速器或代理且网络通畅  
> 如果是使用安卓设备+有线ADB运行 Gakumas Assistant，则需关闭ADB安全设置(否则无法获取屏幕或无法点击)

如果是打算在PC(DMM)上使用 Gakumas Assistant，请保证系统环境有剩余的性能完成实时图像推理  
> DMM模式下Gakumas Assistant会自动申请管理员权限用于屏幕点击  

此外，**必须关闭汉化插件**，当前版本仍未支持汉化版本。

## 命令行模式
如果只想通过命令行使用程序，可以运行根目录下的 `app.py`，它不会启动 WebUI 或 HttpAPI，也不是 TUI 界面。

常用命令：
```bash
python app.py --cli status
python app.py --cli tasks list
python app.py --cli tasks run
python app.py --cli tasks run auto_purchase
python app.py --cli tasks run --from dispatch_work
python app.py --cli config get base.run_mode
python app.py --cli config set base.adb_connect_mode USB
python app.py --cli config export -o config.backup.json
python app.py --cli config import config.backup.json
python app.py --cli resources status
python app.py --cli resources check
python app.py --cli resources apply
python app.py --cli adb devices --usb
```

补充说明：
- `tasks run` 不传任务 ID 时执行全部已启用的自动任务。
- `tasks run <任务ID>` 只执行单个任务。
- `tasks run --from <任务ID>` 从指定任务开始执行后续自动任务。
- `status` 默认不会初始化设备和模型；需要完整状态时使用 `status --full`。
- 需要脚本集成时给命令加 `--json`，输出会使用 JSON 格式。
