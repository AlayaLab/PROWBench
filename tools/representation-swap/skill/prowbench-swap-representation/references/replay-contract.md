# 回放数据与验证边界

## 输入

针对 materialized 的 `generative-scene.v1`，原记录一般包含：

| 文件 | 回放用途 |
| --- | --- |
| `manifest.json` | 坐标约定、实体 ID、GLB 节点映射、静态初始状态 |
| `scene/scene.glb` | canonical 几何，节点以 `userData.nodeId` 映射到记录 |
| `scene/frames.jsonl.gz` | 每帧 index/time、动态节点、实体 worldMatrix、joints 与交互记录 |
| `cameras/cameras.json` | 相机轨迹索引，引用相对该索引文件的位置 |
| `cameras/<camera>/camera.json` | 逐帧 cameraToWorld、projection、内参及可选 hiddenEntityIds |

`frame.entities` 可能只包含动态实体。其他实体从 `manifest.initial.entities` 读取；不能把缺少动态更新视为对象消失。

坐标：Y 向上、米制、右手系、xyzw 四元数、列主序矩阵、相机朝 -Z。不要只用 fov/aspect 重建镜头；直接恢复完整的 projection，保留非对称视锥等信息。

下载版可能仅发布一个视角，但相机索引仍列出其他视角。`prepare_episode.py` 把副本的索引限制为请求的相机，避免回放器请求不存在的 track。原相机数组不改动。

原记录经过 materialization 后，历史 provenance/sourceHashes 可能与当前字节不同。准备阶段对实际读取的 manifest、状态、GLB、相机索引、选中相机重新计算 SHA256；回放结束再次检查。

## 显示对象

运行时先 `seek(frameIndex)` 恢复 canonical 节点。custom adapter 的 `update` 接收只读含义的 state/frame/nodes；以原 worldMatrix 驱动一个独立 Group。不要写入记录 JSON、canonical 节点、源相机数组，也不要重新求解动作或物理。

示例中人物总高与半径是固定局部几何。更换资产的朝向、尺寸或中心点，在子对象的局部变换中处理；保持 Group 的世界变换与原实体一致。几何改变会改变轮廓、遮挡与像素标注，不应声称这些仍与原表示相同。

记录的 `visible` 与相机策略 `hiddenEntityIds` 都需要保留。回放器在实际绘制时临时应用相机隐藏策略，随后恢复对象可见性。

## 运行环境

前端代码被打包在 `assets/runtime/`，只依赖声明的 npm 包。Three.js 负责 GLB 与 WebGL2，Vite 提供本地模块服务，Playwright 控制 Chromium headless shell。`.gz` 通过专门中间件以原始字节提供，浏览器只解压一次。

NVIDIA 容器的 Vulkan/EGL 注册 JSON 指向已有驱动库，不安装或替换系统驱动。成功报告必须记录实际场景画布的 NVIDIA renderer；仅观察 `nvidia-smi` 或浏览器启动成功不够。软件设备不能自动冒充 GPU 结果。

## 校验

1. 每个 frame 的 source state 在绘制两种表示前后序列化一致。
2. 浏览器回放检查 canonical 节点、关节与实体根相对记录的误差；数值四元数 angleTo 可出现约 1e-7 的浮点误差，阈值为 1e-4。
3. 导出两次实际相机 world/projection 和所有实际实体根矩阵。Python 从原始文件重新读取对应值，独立比较，容差 1e-10。
4. 单独比较替换代理 Group 的实际 worldMatrix 与原实体 worldMatrix；局部顶点和局部子对象矩阵跨帧应固定。
5. 所有源文件 SHA256 前后不变；已解码视频帧数、fps、尺寸与时长正确。
6. 代表 PNG 的像素差证明新表示实际绘出；比较位置与遮挡需人工看图。若目标整个片段被遮挡或被第一人称策略隐藏，像素变化检查会失败，应选择可见目标做示例。

`maxContactError` 来自 canonical 骨架与原记录，不证明新代理的可见表面接地。RGB 修改不自动生成新 depth/normal/ID，也不等于重新运行视频生成模型。

## 已验证命令的参数

样本 `city-crossing` / follow / player；150 帧、30 fps、1280×720；胶囊高 1.78、半径 0.26、颜色 `#ffcc22`。两套视频均从同一记录重新渲染，因此对照不受历史视频的编码器或 renderer 版本变化干扰。

准备和验证脚本只依赖 Python 标准库，制作对照图额外使用 Pillow；验证要求 FFmpeg 同时提供 ffprobe、libx264 和 drawtext。图形导出由同一渲染流程交替生成两种表示，视频编码使用 CPU。
