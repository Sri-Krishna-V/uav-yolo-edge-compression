# Benchmark platforms: install and run

`scripts/studybench.py` runs on the platform. It times every test-dev image (pre, net and post separately, batch 1)
and saves the detections. `scripts/studyscore.py` scores them later on the laptop. `scripts/studyexport.py` makes
the artifacts on the laptop, except TensorRT engines, which the Jetson builds itself.

The bench needs only Python, numpy, OpenCV and the one runtime. The commands below install exactly that.

## Raspberry Pi 5 and Pi 4B (Raspberry Pi OS 64-bit, Bookworm, Python 3.11)

From the bench build of 2026-10-21 (ed5cc9f). The wheels resolve for cp311 aarch64. **Not yet run on a Pi.**

```bash
python3 -m venv ~/sb && . ~/sb/bin/activate
pip install numpy==2.4.6 opencv-python-headless==5.0.0.93 onnxruntime==1.31.0 openvino==2026.4.1
pip install --no-deps ncnn==1.0.20260526   # --no-deps: ncnn otherwise pulls in opencv-python, which clashes with the headless build
```

## Galaxy M34 (Termux, Python 3.14)

From the same build. Versions seen in the Termux index on 2026-10-21: python 3.14.8, python-numpy 2.5.4,
python-onnxruntime 1.31.0-3, opencv-python 4.14.0-4 (it pulls in qt6, a large download). **Not yet run on the phone.**

```bash
pkg install x11-repo && pkg install python python-numpy python-onnxruntime opencv-python
```

## Jetson Orin Nano (JetPack 6 only)

The rule (2026-10-21, narrowed 2026-10-22): an Orin Nano on JetPack 6 whose software installs in one day gets a
small column. Both study models; the baseline configuration as a TensorRT FP16 engine and the quantized configuration
as a TensorRT INT8 engine; input resolutions 640 and 960; latency only. Anything else, JetPack 5 included, is cut and
listed as future work. Power state, chosen by default and recorded in every run: the highest nvpmodel mode the board
offers, with `jetson_clocks` on.

Sources: Ultralytics' [NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) ("Run on JetPack
6.1": the torch, torchvision and cuDSS lines), NVIDIA's
[PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
(numpy 1.26 before the torch wheel), NVIDIA's wheel index `developer.download.nvidia.com/compute/redist/jp/` and
`pypi.jetson-ai-lab.io/jp6/cu126` (the same torch 2.10.0 / torchvision 0.25.0 builds). All checked 2026-10-22.

### What the board needs from the laptop

| What | Size | Why |
|---|---|---|
| This repo (`git clone`) | small | the scripts and `cfg/` |
| `datasets/VisDrone/images/val` + `labels/val` (548 each) | 82 MB | INT8 calibration. Without them Ultralytics would start VisDrone's 2 GB download, so `studyexport.py` stops first |
| `datasets/VisDrone/images/test` (1,610) | 301 MB | the bench |
| `best.pt` of `yolo11s-640-s0`, `yolo11s-960-s0`, `yolo26s-640-s0`, `yolo26s-960-s0` | 4 x ~19 MB | the four exports. YOLO26s comes from Vidhyarth's Kaggle output |

On the laptop (Git Bash), make two tars and get them to the board by USB stick, `scp` or a shared link:

```bash
cd "C:/Users/srikr/Desktop/Studies/Minor/Term 5/study/datasets"
tar -cf visdrone-jetson.tar VisDrone/images/val VisDrone/labels/val VisDrone/images/test
cd C:/Users/srikr/uav-yolo-checkpoints
tar -cf ckpt-jetson.tar yolo11s-640-s0/best.pt yolo11s-960-s0/best.pt   # add yolo26s-640-s0/best.pt yolo26s-960-s0/best.pt
```

Keep each `best.pt` in a folder named after its run: `studyexport.py` takes the run name from that folder.

### The command sequence

**1. Probe** (read-only, no sudo; the clone needs this guide's commits pushed). Paste the whole output into
issue #11. Stop if the verdict is CUT. If it is BLOCKED, fix each blocker it lists first.

```bash
git clone https://github.com/Sri-Krishna-V/uav-yolo-edge-compression.git ~/uav
python3 ~/uav/study/scripts/jetson_probe.py | tee ~/jetson_probe.txt
```

**2. Power mode.** Use the mode id the probe prints on its `run mode` line. `jetson_clocks` does not survive a reboot,
so run it again after every boot.

```bash
sudo nvpmodel -m <id>     # it may ask to reboot
sudo jetson_clocks
```

**3. Install** (JetPack 6.1 or 6.2: L4T R36.4, Python 3.10, CUDA 12.6, TensorRT 10.3). The venv sees the system
packages, because JetPack's TensorRT is an apt package (`python3-libnvinfer`) outside any venv.

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv     # skip if the probe shows pip and venv
python3 -m venv --system-site-packages ~/uav-venv && . ~/uav-venv/bin/activate
pip install -U pip
pip install numpy==1.26.4    # NVIDIA's PyTorch-for-Jetson guide pins numpy 1.26; a precaution for JetPack's TensorRT bindings
U=https://github.com/ultralytics/assets/releases/download/v0.0.0
pip install $U/torch-2.10.0-cp310-cp310-linux_aarch64.whl $U/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
# cuDSS, which torch 2.10 needs (Ultralytics' Jetson guide)
D=https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers
wget $D/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get -y install cudss
# the study pins that apply here (requirements.txt and the toolchain row of INVARIANTS.md section 1.3)
pip install ultralytics==8.4.149 pyyaml==6.0.3 onnx==1.23.2 onnxslim==0.1.98 opencv-python==4.11.0.86 numpy==1.26.4
python -c "import torch, torchvision, tensorrt, cv2, numpy, ultralytics; print(torch.__version__, torch.cuda.is_available(), torchvision.__version__, tensorrt.__version__, cv2.__version__, numpy.__version__, ultralytics.__version__)"
```

The last line must print `2.10.0 True 0.25.0 10.3.x ... 1.26.4 8.4.149`. torch goes in before ultralytics, so pip
does not fetch a PyPI torch first. The torch version differs from the study's 2.14.0: the Jetson only exports and
benchmarks, and the Jetson wheels exist only for these versions. The manifest and the run record it.

- *No sudo:* cuDSS also comes as a tarball. Unpack
  `https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-aarch64/libcudss-linux-aarch64-0.7.1.4_cuda12-archive.tar.xz`
  in `~` and add `export LD_LIBRARY_PATH=$HOME/libcudss-linux-aarch64-0.7.1.4_cuda12-archive/lib:$LD_LIBRARY_PATH`
  to `~/uav-venv/bin/activate`. Without sudo you also cannot set the power mode; the runs then record the mode they
  ran in. Not tried.
- *JetPack 6.0 (L4T R36.3: TensorRT 8.6, CUDA 12.2):* the torch 2.10 wheel does not fit CUDA 12.2. Use
  `$U/torch-2.3.0-cp310-cp310-linux_aarch64.whl` and `$U/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl`
  instead, and skip cuDSS. The runner's TensorRT calls exist in 8.6 too. Not tried.

**4. Copy the files** into the clone:

```bash
mkdir -p ~/uav/study/datasets ~/uav-ckpt
tar -xf visdrone-jetson.tar -C ~/uav/study/datasets
tar -xf ckpt-jetson.tar -C ~/uav-ckpt
```

**5. Export**: four engines per study model, FP16 and INT8 at 640 and 960. Each build takes minutes; INT8 first
calibrates on the 548 val images.

```bash
cd ~/uav/study && . ~/uav-venv/bin/activate
for run in yolo11s-640-s0 yolo26s-640-s0; do
  python scripts/studyexport.py --weights ~/uav-ckpt/$run/best.pt --imgsz 640 --only tensorrt-fp16 tensorrt-int8
done
for run in yolo11s-960-s0 yolo26s-960-s0; do
  python scripts/studyexport.py --weights ~/uav-ckpt/$run/best.pt --imgsz 960 --only tensorrt-fp16 tensorrt-int8
done
```

Each INT8 manifest records `calibration.images_sha256`. On the laptop's val images it is
`1189ee954c10649f0a00a2a9bebf14107228c0c57b48bf9a374e337392bbf38a`; a different value means different calibration
images.

If a build is killed for memory (most likely INT8 at 960 on a 4 GB board): close the desktop for this boot
(`sudo systemctl isolate multi-user.target`), then rerun that one variant with `--workspace 1`. The manifest records
the workspace.

**6. Bench**: eight cells, every test-dev image. Leave the board idle while it runs.

```bash
for cell in yolo11s-640-s0/640 yolo26s-640-s0/640 yolo11s-960-s0/960 yolo26s-960-s0/960; do
  for v in tensorrt-fp16 tensorrt-int8; do
    python scripts/studybench.py --artifact exports/$cell/$v --runtime tensorrt --imgsz ${cell#*/} \
        --images datasets/VisDrone/images/test --out bench/jetson/$cell/$v --platform-tag jetson
  done
done
```

The bench prints a WARNING if the board is not in its highest mode with `jetson_clocks` on. A run that dies resumes
with the same command plus `--resume`.

**7. Bring home** the runs and the manifests (not the engines: they run only on this board):

```bash
cd ~/uav/study && tar -cf ~/jetson-bench.tar bench/jetson exports/*/*/tensorrt-*/manifest.json
```

On the laptop, unpack it into `study/` and, if wanted, score the detections: `python scripts/studyscore.py
bench/jetson/yolo11s-640-s0/640/tensorrt-fp16 ...`. Accuracy is not required for this column, but it costs nothing.
