# AnyCam

任意相机 RAW + 对应机型 Adobe Standard `.dcp`，在 CUDA GPU 上渲染出四台相机的机内色彩：

| 子命令 | 引擎 | 挡位 |
| --- | --- | --- |
| `fuji` | 富士 X100VI 胶片模拟（kind-13，20 种）+ 标定颗粒 | DR100/200、高光/阴影、色彩、白平衡偏移、CCE / FX Blue、颗粒强弱/大小 |
| `m9` | 徕卡 M9 固件 1.216，BF561 色彩内核 + bf547 自动白平衡 | 对比度 × 饱和度 5×5；AWB / 原片 / 预设 / 63 档开尔文 |
| `m10r` | 徕卡 M10-R 固件 40.26.35.51，CA9 `CC0` + B2Y 内核 | 对比度 × 饱和度 3×3、黑白 |
| `m240p` | 徕卡 M (Typ 240) / M-P 固件 2.1.0.0 | Standard 5×5、Vivid、Natural、黑白 |

## 安装

需要 Python ≥ 3.10、带 CUDA 的 PyTorch、`rawpy`、`tifffile`、`pillow`、`threadpoolctl`。

```powershell
pip install -e .
```

也可以不安装，直接运行仓库里的 `run_anycam.py`。

## 用法

```powershell
python run_anycam.py fuji --list-films

python run_anycam.py fuji --raw DSCF6199.RAF `
  --dcp "C:\ProgramData\Adobe\CameraRaw\CameraProfiles\Adobe Standard\Fujifilm X100VI Adobe Standard.dcp" `
  --film classic_neg --dr 200 --highlight 1 --shadow 2 --color 1 --cce 1 --grain weak --out out.jpg

python run_anycam.py m9 --raw IMG.dng --dcp "OPPO Find X9 Ultra wide camera 23mm f1.5 Adobe Standard.dcp" `
  --wb auto --contrast 1 --saturation 1 --out-space p3 --out out.jpg

python run_anycam.py m10r --raw IMG.dng --dcp X.dcp --wb 4800K,-0.001 --contrast 1 --out out.png
python run_anycam.py m240p --raw IMG.dng --dcp X.dcp --mode vivid --wb camera --out out.tif
```

所有子命令共用：`--raw`、`--dcp`、`--out`（`.png` / `.jpg` 8 位，`.tif` 16 位）、`--out-space {srgb,p3,adobe}`、`--ev`、`--denoise {off,low,medium,high}`、`--no-hsm`、`--half-size`、`--device {auto,cuda,cpu}`。各引擎的挡位见 `python run_anycam.py <子命令> --help`。

## 管线

```
RAW ─ rawpy 线性解马赛克（传感器原向）─ [降噪] ─ 引擎 ─ 输出编码 ─ 按 EXIF 转正 ─ [富士颗粒] ─ 单张导出
```

### RAW 解码（`anycam/core/raw.py`）

- `rawpy.postprocess`：`gamma=(1,1)`、`no_auto_bright=True`、`user_wb=[1,1,1,1]`、`output_color=raw`、`adjust_maximum_thr=0`、`user_flip=0`，不用 LibRaw 或 RAW 内嵌的颜色矩阵。M9 的稀疏 AWB 格点按传感器原生坐标取样；导出前（颗粒之前）用 `orient_np` 转正，所以富士颗粒的 48 µm 尺度与原向出图一致。
- LibRaw 的 OpenMP X-Trans 插值有线程竞争（同一张 RAF 重复解码会有像素不同），X-Trans 解码固定单线程；Bayer 与线性 DNG 照常多线程。
- 曝光基线：DNG 读 `BaselineExposure`；富士 `.RAF` 读 DR100/200/400。徕卡三引擎在场景 18% 灰口径下，`.RAF` 基线为 `log2(15360/3200) − 1 + log2(0.569589) + dr_ev`（DR100 约 +0.451 EV）；富士引擎的前段自带 `0.569589` 输入缩放，`.RAF` 基线为 `log2(15360/3200) − 1 + dr_ev`（DR100 约 +1.263 EV）。两者由 `RawFrame.raw_base_ev(raf_input_scale=…)` 显式区分。

### 降噪（`--denoise`，`anycam/core/denoise.py`）

在线性相机 RGB 上、进入任何引擎之前执行，M9 AWB 看到的也是降噪后的数据。噪声水平从画面本身测：相邻像素差按亮度分档，用 MAD 稳健估计拟合 `σ² = a·x + b`，强度随 ISO 与曝光自动变化。亮度 `Y = (R+2G+B)/4` 走噪声自适应 5×5 双边滤波；色度 `R−G`、`B−G` 以降噪后的亮度为引导做引导滤波，去彩噪远强于去亮噪，保住纹理和边缘。默认关闭。

### 同源双 DCP 对接（徕卡三引擎，`anycam/core/bridge.py`）

源机 Adobe Standard 的 `ForwardMatrix + HueSatMap` 把画面送进线性 ProPhoto D50，再反解目标机 Adobe Standard 的 `HueSatMap + ForwardMatrix` 回到目标机 raw；两端出自同一标定者，交接白点相同，不使用 `ProfileToneCurve`。整帧渲染用预建的二维逆 HSM 表做起点加两步迭代，M9 AWB 循环用 24 步纯迭代。

- **M9**：`BE_M9 = −0.5`。`--wb auto` 在传感器原向 20×30 阵列的内区 19×29 上按 4×4 稀疏小窗（B 通道左错一格）取样，代入 bf547 AWB（`assets/m9/awb_p3.bin`），从 D65 起步迭代到不动点；另有原片白平衡、固件预设、63 档开尔文表。
- **M10-R**：`BE_M10R = 0`，白电平 15000。`--wb camera`：源机 Adobe `CM` 求场景白点 → M10-R Adobe `CM` 求增益 → 固件 `CM_LT / CM_HT` 求 `CCT_fw` 驱动 `CC0`。`--wb <CCT>[K][,<Duv>]`：先由固件 `CM_LT / CM_HT` 得增益与 `CC0`，再用 Adobe 侧 `white_of_neutral(1/gain)` 统一交接白点。
- **M240**：`BE_M240 = 0`，14 位白电平 15000 右移 2 位进 12 位 `colorcomp`（`3750`）。场景白点、增益、对接全部用 M240 Adobe Standard 的 `CM / FM / HSM`，不把 Leica DNG 内嵌 `ColorMatrix` 混接到 Adobe `FM / HSM`。

### 富士 X100VI（`anycam/engines/fuji.py`）

源机 DCP 固定用 D65 `ForwardMatrix` 进 ProPhoto D50，帧色温由相机中性点经 `ColorMatrix1/2` 求得；源机 `HueSatMap` 正向、X100VI `HueSatMap` 反向对齐传感器色彩，然后是完整 kind-13 管线：`FRONT_M` → 白平衡偏移 → 1:2:1 亮度 → tone40 → 胶片矩阵 → tables31 → chroma4 → ±0.5 钳位 → gmod → 17³ 3D LUT → huegain / hueres → CCE / FX Blue。该引擎保留自己的 HSM 内核（支持三维表、sRGB 编码表用标准 sRGB 曲线、V 钳到 1、16 轮迭代反解），它们属于已标定的观感，与对接用的二维内核不同。

X100VI HSM 的 16 轮反解按帧预建成表：二维 HSM 下，H/S 的迭代在 V 不触顶时与 V 无关，所以在 V = 0.5 上把“16 轮后的色相偏移、饱和度、V 比例”按 (h, s) 列表，逐像素查表。16 轮在部分颜色上并未收敛、映射不平滑，建表时在每格中心与精确反解比对，超出容差的格子，以及可能触发 V 钳位的亮像素和近黑像素，仍走 16 轮原迭代。成片与逐像素 16 轮相比 MAE ≈ 1e-7、最大差 < 4e-4，渲染约快 2 倍；`render_image_gpu(..., exact_inverse=True)` 可回到全帧 16 轮。颗粒（`fuji_grain.py`）按 X100VI 全幅对角线锚定 48 µm 孔径，跨分辨率尺度不变。

## 资产

`assets/` 只放查找表：`fuji/x100vi_*.bin`（20 个）、`m9/leica_m9.bin` 与 `m9/awb_p3.bin`、`m10r/leica_m10r.bin`、`m240p/leica_m240p.bin`，共约 3.2 MB。
