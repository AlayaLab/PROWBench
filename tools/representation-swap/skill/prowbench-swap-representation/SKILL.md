---
name: prowbench-swap-representation
description: "从已下载的 PROWBench 场景记录替换可见代理几何或材质，保留原相机和物体轨迹，导出配对视频与逐帧校验。用于同轨迹的 representation 重渲染；不负责视频模型推理或评测。"
---

# PROWBench 同轨迹替换 representation

复用已经录好的世界状态，替换显示 adapter。随 skill 提供一个可运行的例子：在 `mixed` 场景中将指定人物替换为黄色胶囊，其他实体沿用 mixed。用户指定的几何、配色、视角和输出范围优先于例子。

## 选择输入与范围

- 使用下载数据里的 episode 根目录，包含 `manifest.json`、`scene/scene.glb`、`scene/frames.jsonl.gz`、`cameras/`。有 `versions/` 的 case 应选择实际含 manifest 的版本目录。
- 固定 episode、camera、实体 ID、帧号和 fps。相同 episode 的不同 camera 具有不同相机轨迹。
- 若用户只切换已有 `white/mixed/colored-obb` 视频，直接选同一 `views/<camera>/` 下对应文件与 prompt，无需重渲染。
- 自定义表示使用下面的回放路径。选择镜头内可见的目标；第一人称镜头常隐藏 `player`，需查看 camera 的 `hiddenEntityIds`。
- 少量预览用 `--frames`；整段视频省略它。先看预览有助于确认 pivot 和形状，但不把预览变成已授权整段输出的审批环节。

## 运行入口

以本文件所在目录的绝对路径设置 `$SKILL`，以原始 episode 目录设置 `$EPISODE`，以新的输出目录设置 `$WORK`。这些变量由当前环境决定，不能依赖其他工作区的绝对路径。

脚本需要 Node.js、Three.js 0.181.0、Vite 7.1.5、Playwright 1.56.1、FFmpeg/ffprobe，以及 Python 3 + Pillow。推荐 Node 22.12+，满足 Vite 的版本要求。

新环境在 skill 目录运行 `npm ci`，再运行 `npx playwright install chromium`；Linux 缺少浏览器系统依赖时使用 `npx playwright install-deps chromium`。Python 验证脚本所在环境需有 Pillow。skill 已附必要回放代码，不依赖原 coastal 工程，也不依赖其他位置的 `node_modules`。使用 Playwright 默认浏览器缓存，或显式设置 `PLAYWRIGHT_BROWSERS_PATH` 复用已有缓存。

### 1. 准备只读回放输入

```bash
python3 "$SKILL/scripts/prepare_episode.py" \
  --episode "$EPISODE" --camera follow --entity player \
  --out "$WORK/replay-input"
```

输出目录须为空。脚本只在新目录改写资源路径，并过滤相机索引，GLB、状态流和选中相机指向原文件。保存实际文件 SHA256；不要用 `sample.json.sourceHashes` 的历史哈希代替当前输入哈希。

### 2. 替换显示 adapter

修改 [assets/replacement.js](assets/replacement.js) 中的 factory；黄色胶囊可直接用，无需改代码。参数 `--entity`、`--color`、`--height`、`--radius` 控制对象与几何。

新 adapter 的约定：

```js
return {object, update({state, frame, nodes}) {
  object.matrix.fromArray(state.worldMatrix);
  object.visible = state.visible;
}};
```

`object.matrixAutoUpdate=false`，显示对象独立于 canonical 节点。复制 `replay.modes.mixed` 配置，仅覆盖需要替换的 entity 或 semantic 映射；保留相机、背景与其他实体的配置。给新表示单独命名，避免把自定义表示误标为官方 mixed。

胶囊例子的 Y 轴向上，底端在实体局部原点，总高 1.78 m、半径 0.26 m。新资产应明确本地轴、尺度与 pivot，通过局部子对象偏移对齐；不要修改记录的根变换来补偿模型偏移。固定代理不应随逐帧 bbox 拉伸。若需要关节动画，显式使用记录的骨架驱动可见关节。

### 3. GPU 预览与整段输出

```bash
node "$SKILL/scripts/render_pair.mjs" \
  --input "$WORK/replay-input" --out "$WORK/preview" \
  --frames 0,75,149

node "$SKILL/scripts/render_pair.mjs" \
  --input "$WORK/replay-input" --out "$WORK/render" \
  --entity player --color '#ffcc22' --height 1.78 --radius 0.26
```

预览帧号按实际长度选择，例子为 150 帧。各输出目录须为空；新几何版本使用新目录。脚本在同一进程中交替绘制同帧的 mixed 和自定义代理，输出两个无 UI 的 MP4、代表 PNG、逐帧矩阵与设备记录。端口冲突时传 `--port <空闲端口>`。

GPU 为默认，必须核实真实场景画布的 unmasked WebGL renderer；GPU 失败会退出。用户明确选择软件渲染时才传 `--render-backend cpu`。`PLAYWRIGHT_BROWSERS_PATH` 可指向既有浏览器缓存，`FFMPEG` 可覆盖编码器路径。H.264 使用 CPU libx264，不能称为全流程 GPU 编码。

### 4. 独立验证与交付

```bash
python3 "$SKILL/scripts/verify_pair.py" \
  --input "$WORK/replay-input" --render "$WORK/render" --out "$WORK/checks"
```

验证器重新读取原记录，比较两套图像对应的每帧相机矩阵、投影矩阵、所有实体根矩阵及替换代理矩阵；核对源文件 SHA256，完整解码 MP4，并制作对照图和左右并排视频。`checks` 也使用空目录。

还要实际查看代表帧：新几何是否出现在正确位置、是否贴近支撑面、遮挡是否合理、其他对象是否受到误改。原骨架的 contact 验证不能代替新几何接地检查。

交付 MP4、对照图、`checks/verification.json`、日志和绝对路径。说明这一实验验证控制输入的同轨迹渲染；不据此保证生成模型的输出轨迹。新 representation 的 depth/ID/normal 必须使用新几何另行导出，本例只输出 RGB。

## 参考与已验证示例

- 输入格式与独立校验细节：[references/replay-contract.md](references/replay-contract.md)。处理其他布局、相机或新 adapter 时读取。
- 回放代码来源与哈希：[references/runtime-provenance.json](references/runtime-provenance.json)。升级运行时代码时保留来源记录并重跑配对验证。
- 已验证样本：`verified_first_frame/city-crossing`，follow，player，150 帧 / 30 fps / 1280×720。独立示例包的 `README.md` 提供安装与复现命令，`results/city-crossing-capsule/` 提供完整视频与验证证据。单独安装 skill 后可按上面的命令生成新结果。
