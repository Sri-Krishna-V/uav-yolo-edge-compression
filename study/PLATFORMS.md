# Benchmark platforms: install and run

`scripts/studybench.py` runs on the platform. It times every test-dev image (pre, net and post separately, batch 1)
and saves the detections. `scripts/studyscore.py` scores them later on the laptop. `scripts/studyexport.py` makes
the artifacts on the laptop, except TensorRT engines, which the Jetson builds itself. `scripts/studysession.py` runs
a whole platform session through the bench (see "Running a session").

The bench needs only Python, numpy, OpenCV and the one runtime. The commands below install exactly that.

## Raspberry Pi 5 and Pi 4B (Raspberry Pi OS 64-bit, Bookworm, Python 3.11)

From the bench build of 2026-10-21 (ed5cc9f). The wheels resolve for cp311 aarch64. **Not yet run on a Pi.**

```bash
python3 -m venv ~/sb && . ~/sb/bin/activate
pip install numpy==2.4.6 opencv-python-headless==5.0.0.93 onnxruntime==1.31.0 openvino==2026.4.1
pip install --no-deps ncnn==1.0.20260526   # --no-deps: ncnn otherwise pulls in opencv-python, which clashes with the headless build
```

### The Pi 5 wizard

The Pi 5 is with us until 31 October (#10), so it keeps its own SD card and its install. `pi/wizard.sh` takes one
plan through a Pi session from the laptop (Git Bash), with no agent in the loop:

```bash
cd study
bash pi/wizard.sh plans/pi5-speedtest.json    # first: is INT8 slower than FP32 on the Pi 5? (#25)
bash pi/wizard.sh plans/pi5-fullcell.json     # then: the real time of one full cell
bash pi/wizard.sh plans/pi5-energy-check.json # the energy mode's first real run (#26; "Energy sessions" below)
```

**A blank SD card first:** `bash pi/provision.sh` (9 stages, about 30 minutes) has Raspberry Pi Imager write
Raspberry Pi OS (Legacy, 64-bit) Lite. That is Bookworm with Python 3.11, the OS these pins are for. Imager's first
entry, Raspberry Pi OS (64-bit), is Trixie with Python 3.13. The wizard also writes this laptop's SSH key onto the card,
and the network to join is the laptop's Mobile Hotspot or a home Wi-Fi, because the campus Wi-Fi cannot be joined by
the Pi. After the first boot, the wizard connects to the Pi and checks the board, the OS, the fan, the power, the SD
card and the internet. It records them in `verification/pi5_facts.txt`, writes the host and user to `pi/.env`, and
hands over to `pi/wizard.sh`.

**Have ready:** the Pi 5 on its official 27 W supply, with its fan's 4-pin plug in the FAN socket and `arm_freq=2000`
in `/boot/firmware/config.txt` (the clock lock, INVARIANTS.md §4.7), on the laptop's network; its host name or IP and
user; SSH on (`raspi-config` → Interface Options → SSH); the Pi's password (for the SSH key and the
sudo fixes); internet on the Pi if the install is not done yet.

Its 7 stages: what to have ready; connect (it sets up an SSH key if there is none); fan, power and SD card; copy the
scripts, the plan, the exports and the test-dev images (a path already on the Pi with the same content is not copied
again); the install and the setup checks (it offers to run the fixes); start the session under `nohup`, or follow or
resume one (starting or resuming, it asks for the room's temperature, which the session records); copy the results
home to `study/bench/` and score them. Every stage finds what is already done, so run it
again at any time: after closing it, after a dropped connection, or when a session finishes. Your host and user are kept
in `pi/.env` (not in git).

**Time,** measured at the 2000 MHz lock on 26–27 October ([pi5_pilot_record.md](verification/pi5_pilot_record.md),
[pi5_speedtest_record.md](verification/pi5_speedtest_record.md)). YOLO11s A1 at 640, median latency per image:
ONNX Runtime FP32 390 ms, ONNX Runtime INT8 272 ms, OpenVINO FP32 294 ms, NCNN (FP16) 171 ms.

- the speed test: 26 minutes, gates included;
- the full cell (ONNX Runtime FP32, all 1,610 images): 11 min 48 s for the cell with its warm-up, and 21.5 minutes for
  the session with the reference runs and the gates;
- a new plan: each full cell takes about 1,630 × its latency, plus a start gate of up to 3 minutes; each session adds
  about 8 minutes for its two reference runs and their gates. At 960 and 1280, expect about 2.25× and 4× the 640
  latency (Ultralytics' scaling, not yet measured here).

The speed test's NCNN cell runs the `ncnn-fp32` export, but NCNN computes in FP16 on the Pi 5 (#13): its row is NCNN
FP16, the Pi 5's NCNN baseline, and not an FP32 against INT8 pair.

**Rehearsal.** `bash pi/tests/rehearse.sh` runs the whole wizard with the laptop standing in for the Pi (a fake ssh,
`plans/laptop-dry-run.json`, `--dry-run`), through the copy home and the scoring, into a temporary folder. On
2026-10-22 it passed in about 5 minutes, with a second run that copied nothing and found the session complete.

## Galaxy M34 (Termux, Python 3.14)

From the same build. Versions seen in the Termux index on 2026-10-21: python 3.14.8, python-numpy 2.5.4,
python-onnxruntime 1.31.0-3, opencv-python 4.14.0-4 (it pulls in qt6, a large download). **Not yet run on the phone.**

```bash
pkg install x11-repo && pkg install python python-numpy python-onnxruntime opencv-python
```

## Running a session

`scripts/studysession.py` runs one platform session: every cell of a **platform plan**, each through `studybench.py`,
under the platform-run rules decided on #8 (2026-10-22). Run it on the platform, from `study/`, inside `tmux` or
`nohup` so that a dropped SSH connection does not stop it:

```bash
python3 scripts/studysession.py plans/<plan>.json --out bench/<platform>/<session>
```

If it stops anyway, run the same command again. It skips the runs that finished and restarts a half-done one from the
start. Standard library only; it needs no more than the bench does.

### The rules it applies

- **Threads and warm-up:** 4 threads and 20 untimed warm-up images per run, recorded.
- **Pi setup checks:** on a Pi, before it starts and again on every resume, it checks:
  - the CPU governor is `performance` on every CPU;
  - no desktop session is running;
  - `apt-daily.timer` and `apt-daily-upgrade.timer` are off;
  - the packages the plan needs are at this guide's versions;
  - `vcgencmd measure_clock arm` reads the CPU clock, which the bench logs (#28). Its fix adds the user to the `video`
    group.

  A failed check stops it and prints the fix. `--override-checks "<why>"` runs anyway and records the reason.
- **Start gate**, before every run:
  - a run starts when the CPU is below 50 °C, or after 3 minutes;
  - the phone waits a fixed 3 minutes, and logs its battery temperature when Termux can read it;
  - a Jetson records its power mode and clocks at every gate.
- **Cell order:** shuffled once with the plan's seed, and logged.
- **Drift check:** the reference cell runs first and last: YOLO11s baseline, ONNX Runtime FP32, 640, on the first 200
  test-dev images. If its median latency moves by more than 3%, every cell of the session is flagged for a re-run.
  The reference cell's export and onnxruntime must be on the platform: without them the runner does not start.
- **Crashes:** a crashed cell runs again from the start, once. A second crash marks it failed.
- **Throttling:** a throttled cell runs again at the end of the session, and both attempts are kept.
  - Pi: `vcgencmd` throttle flags.
  - Jetson: over-current counters and the GPU clock.
  - The laptop and the phone give no signal, so their throttle state is "unknown".
- **Latency:** the bench's `e2e_deploy` median: pre + net + post at the deployment threshold, 0.25 (#22,
  INVARIANTS.md §4.5). A bench from before #22 gives no `e2e_deploy`. For one of those the runner uses pre + net + post
  at the evaluation threshold, and says so in the log.

### The plan

A plan is a JSON file in `plans/`:

```json
{
  "platform": "pi5",
  "note": "the speed test",
  "seed": 0,
  "cells": [
    {"artifact": "exports/yolo11s-640-s0/640/onnxruntime-int8", "runtime": "onnxruntime", "imgsz": 640, "limit": 200},
    {"artifact": "exports/yolo11s-640-s0/640/ncnn-fp32", "runtime": "ncnn", "imgsz": 640}
  ]
}
```

| Key | Meaning |
|---|---|
| `platform` | required: `pi5`, `pi4b`, `m34`, `jetson` or `laptop` |
| `cells` | required. Each cell has `artifact` (a `studyexport.py` directory), `runtime` and `imgsz`. Optional: `limit`, the first N test-dev images (leave it out for a full cell), and `name`, which defaults to the artifact path after `exports/` (give one when two cells share an artifact) |
| `seed` | the shuffle's seed; default 0 |
| `note` | free text, logged |
| `images` | default `datasets/VisDrone/images/test` |

A misspelt key, or a missing artifact or image directory, stops the runner before anything runs.

### What it writes

In `--out`:

- `session.log`: every decision.
- `session.json`: the state it resumes from. Each invocation's record holds the platform's facts and `ambient_c`, the
  room's temperature given with `--ambient-c` (the Pi wizard asks for it each time it starts or resumes a session;
  `null` when none was given). On a Pi, the facts include `cooling` (#28):
  - `fan_rpm`: the fan on the Pi 5's fan connector. `"none"` when nothing is there. `null` (unknown) on a Pi 4B, which
    cannot see a fan, or when the fan's reading fails;
  - `arm_freq`, `arm_freq_min` and `temp_soft_limit`, from `vcgencmd get_config`. Raspberry Pi's docs list
    `temp_soft_limit` for the 3A+ and 3B+ only, so on a Pi 5 its value may mean nothing;
  - `idle_temp_c`: the CPU temperature before the invocation's first run. On a resume right after a run, the CPU may
    still be warm.
- `summary.csv`: one row per cell and per reference run, with attempts, outcome, median latency, throttle state, the
  drift verdict, and `run_dir`, the attempt to score with `studyscore.py`.

Each attempt gets its own `runs/<cell>/attempt-<n>/`, holding the bench's files and its output in `bench.log`.

**The CPU clock** (#28). On a Pi, the bench's `cpu_mhz` column is the clock the firmware runs the cores at, from
`vcgencmd measure_clock arm`, read after each image's timing. Linux's `scaling_cur_freq` does not see a firmware
throttle: under the `performance` governor it says 2400 MHz on a Pi 5 the firmware has cut to 1.5 GHz. Other platforms
keep `scaling_cur_freq`. `meta.json`'s `cpu_clock` names the source, gives what one `vcgencmd` call cost (`call_ms`:
median and max over this process's calls, written every 50 images and when a run stops), and counts the reads that
failed. `device_state.cpu_mhz` gives the clock's min, median and max. An energy run reads the clock every 50 frames,
with the throttle flags.

- **Cost of one `measure_clock` call on the Pi 5:** a median of 11.6–12.8 ms in each of the 13 runs of 26–27 October,
  and at most 116 ms (`cpu_clock.call_ms`). The call runs after the image's timing, so no latency includes it; it adds
  about 20 s to a full cell. The clock is still read on every image.

`--needs` prints the paths a plan needs on the platform, and `--check` runs only the setup checks (exit 1 if one
fails). Neither runs a cell or writes anything; the Pi wizard uses both.

### Energy sessions (Pi 5 only)

A plan with `"mode": "energy"` is an energy session (#26, INVARIANTS.md §4.6). It is never mixed with latency: an energy
session times no latency, and a latency session never runs the PMIC sampler. Its cells take no `limit`, and it runs no
reference cell.

- **Each cell:** the start gate; a 1-minute idle reading of the PMIC rail sum; then 3 minutes of the cell over the first
  100 test-dev images (decoded beforehand, cycled) with the PMIC read every 0.5 s.
- **`sampler-off`, right after the plan's first cell:** that cell once more, with the same rest and 3 minutes, and the
  sampler off. This measures the sampler's cost on FPS. One pair of runs bounds the cost rather than resolving it.
- **The setup checks** add one: `vcgencmd pmic_read_adc` gives a rail sum. If it does not, the session refuses to start.
- **`summary.csv`:** FPS, the idle and run rail sums (W), J per frame and J per frame net of idle, all labelled
  "rail sum, not wall power". `session.json` holds the sampler's cost. Each run's `power.csv` keeps every reading with
  its window.

`plans/pi5-energy-check.json` is the first real one: A1 at 640, ONNX Runtime FP32 and INT8, about 15–25 minutes. It
checks the energy mode on the board. The equal-latency pairs get their own plan once the Pi 5's latencies give the pairs.
`bash pi/wizard.sh plans/pi5-energy-check.json` runs it; the wizard shows the energy summary, and there is nothing to
score.

On the laptop, on 23 October 2026, a dry run of that plan took about 2 minutes, with a fake PMIC reading in place of
`vcgencmd`. The bench's 15 s windows lined up with the session's readings: 20 idle readings and 30 run readings. This
was a plumbing check only. A fake reading costs nothing, so the sampler's cost on FPS can only be measured on the Pi 5.

### Dry run

`--dry-run` caps the gate at 5 s and marks the session DRY RUN; no number from it is reported. In an energy session
it also shortens the idle reading to 10 s and the runs to 15 s.
`plans/laptop-dry-run.json` is a two-cell plumbing check for the laptop. On 2026-10-22 it took 2 minutes, with a peak of
0.85 GB.

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
