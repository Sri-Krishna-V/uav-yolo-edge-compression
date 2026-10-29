# Findings: is the Jetson Orin Nano Super setup and benchmark procedure correct, item by item?

Ticket: #39 (Jetson session: a plan, the drift reference, and a wizard).
Researched: 2026-10-29. Paper due 2026-10-31.

**Question.** A student runs `study/jetson/provision.sh` then `study/jetson/wizard.sh` on the real Jetson Orin Nano Super
Developer Kit tomorrow: JetPack 6.2.1 from the SD card image, first boot, SSH from a Windows 11 laptop (Git Bash,
OpenSSH, hotspot 192.168.137.x), `jetson_probe.py`, `nvpmodel` + `jetson_clocks`, `install.sh` (torch 2.10.0, cuDSS,
pins), `export.sh` (8 TensorRT engines via Ultralytics 8.4.149) and `studybench.py`'s `TensorRT` class. Is each of
those steps true against primary sources, and what must change before tomorrow?

**Verdict in one line:** the plan holds. Every URL resolves with no login, every pip pin has a cp310 aarch64 wheel
(the onnxruntime placeholder `1.23.2` is correct: it is the newest build for Python 3.10), the mode IDs and the
probe's parsers match NVIDIA's own files, and the TensorRT calls exist in 10.3.0. **One real gap:** `install.sh` never
installs `python3-libnvinfer`, and NVIDIA's `nvidia-jetpack-runtime` does not pull it in (only `-dev` does), so
`import tensorrt` may fail on a fresh SD image. Three things to fix or know: the image **expands to 24.09 GB**
(card size), **headless first boot over the USB-C port is real but only documented by NVIDIA for the AGX Orin**,
and YOLO26 **INT8** engines on TensorRT 10.3 are built as one-to-many (the exporter does it by design).

Legend: ✅ = verified against the primary source itself (NVIDIA docs or packages, Ultralytics source, PyPI JSON, the
live URL). ⚠️ = verified only partly, from a secondary page, or not testable without the board. ❌ = does not hold; the
fix is in "Corrections". "Package evidence" below means files I read out of NVIDIA's own apt packages
(`repo.download.nvidia.com/jetson`), version 36.4.4, the same L4T as the 6.2.1 image.

---

## A. OS install

### 1. The 6.2.1 image URL, Imager, card size  ✅ URL / ⚠️ Imager / ✅ card

- ✅ `curl -sIL https://developer.nvidia.com/downloads/embedded/L4T/r36_Release_v4.4/jp62-r1-orin-nano-sd-card-image.zip`
  gives `302` to `https://developer.download.nvidia.com/embedded/L4T/r36_Release_v4.4/jetson-orin-nano-devkit-super-SD-image_JP6.2.1.zip`,
  then `200`, `Content-Length: 11725610175` (11.73 GB, 10.92 GiB). No login. The server accepts `Range` requests, so
  `curl -C -` resumes. The JetPack 6.2.1 page lists this file name under "Jetson Orin Nano Developer Kit"
  ([jetpack-sdk-621](https://developer.nvidia.com/embedded/jetpack-sdk-621)); it says nothing about "Super": the image
  is the `jetson-orin-nano-devkit-super` one.
- ✅ **What is inside.** I read the zip's central directory by HTTP range request (no download): one entry,
  `sd-blob.img`, deflate, **24,086,839,296 bytes (24.09 GB, 22.4 GiB) uncompressed**. So the card must hold at least
  24.09 GB: a 16 GB card fails; a 32 GB card (about 31.9 GB) fits tightly; **NVIDIA says "64GB UHS-1 or bigger
  recommended"** ([Getting Started Guide, Dec 2025 snapshot](https://web.archive.org/web/20251225062626/https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit),
  and the same words in the [JetPack 6.x Update Path](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html)).
  The 11.7 GB zip means a 12 GB download plus nothing else: neither tool needs it unzipped.
- ✅ NVIDIA's documented flasher is **balenaEtcher**; its steps say "Select image, choose the zipped image file"
  (zip as downloaded), "about 15 minutes to write and validate over USB3", and "Windows may let you know it doesn't know
  how to read the SD card: Cancel". Raspberry Pi Imager is not mentioned anywhere on NVIDIA's pages.
- ⚠️ **Raspberry Pi Imager "Use custom".** The Imager source opens every image through libarchive with
  `archive_read_support_filter_all` + `archive_read_support_format_all` + `..._raw`
  ([downloadextractthread.cpp](https://raw.githubusercontent.com/raspberrypi/rpi-imager/main/src/downloadextractthread.cpp),
  lines 385-387), and its image file filter lists `*.img *.zip *.gz *.xz *.zst *.wic`
  ([imagewriter.cpp](https://raw.githubusercontent.com/raspberrypi/rpi-imager/main/src/imagewriter.cpp), line 3506).
  So a `.zip` holding one `.img` is accepted, with no unzipping. I could not test it on this 24 GB image, and NVIDIA
  never tested it. Keep Etcher as the primary and Imager as the fallback, as `provision.sh` already has it. If Imager
  asks for a device, any choice works (a custom image ignores it); say "No" to customisation.

### 2. The JetPack 5.1.3 bridge image  ✅

- ✅ The JetPack 5.1.3 page ([jetpack-sdk-513](https://developer.nvidia.com/embedded/jetpack-sdk-513)) links
  `https://developer.nvidia.com/downloads/embedded/l4t/r35_release_v5.0/jp513-orin-nano-sd-card-image.zip`. `curl -sIL`:
  `302` to `https://developer.download.nvidia.com/embedded/L4T/r35_Release_v5.0/JP513-orin-nano-sd-card-image_b29.zip`,
  `200`, `Content-Length: 8814360870` (8.81 GB). No login. The file name `JP513-orin-nano-sd-card-image_b29.zip`
  is exactly the one NVIDIA's update page demands ("Use the updated file").
- ✅ NVIDIA's [JetPack 6.x Update Path](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html)
  (quoted in full under "What provision.sh must handle"): firmware **older than 36.0** (e.g. 35.x) needs it; firmware
  36.x is ready for JetPack 6. Check with Esc at the NVIDIA logo (UEFI menu shows the version near the top; also works
  on a serial console), or from Linux `sudo nvbootctrl dump-slots-info` ("Current version: 35.5.0" in NVIDIA's example).
  The bridge needs **two required reboots** (after the first firmware update, after installing
  `nvidia-l4t-jetson-orin-nano-qspi-updater`), a **power-off and card swap**, and a **third reboot only if** the
  6.2.1 image schedules another update (`sudo systemctl status nv-l4t-bootloader-config`). It needs internet on the
  JetPack 5 board (`sudo apt update` + `apt install`). "Do not remove power while a firmware update is in progress."
  The page does not say how long it takes.

### 3. First boot (oem-config)  ⚠️ headless / ✅ screens

- ✅ **With a monitor** (what NVIDIA documents for the Orin Nano SD card): DisplayPort monitor, USB keyboard and
  mouse; the kit powers on by itself when the 19 V supply is plugged in; a green LED next to the USB-C port lights.
  First-boot screens listed by the [Getting Started Guide](https://web.archive.org/web/20251225062626/https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit):
  accept the NVIDIA Jetson software EULA; system language, keyboard layout, time zone; connect to a wireless network;
  create username, password and computer name; log in. **The default power mode is "25W"**; switch to MAXN SUPER
  from the NVIDIA icon at the top right of the desktop. (NVIDIA's *current* live guide at the same URL is the JetPack
  7.2.1 ISO flow and no longer applies; JetPack 7 has no SD image.)
- ✅ **Package evidence** (`nvidia-l4t-oem-config_36.4.4-20250616085344_arm64.deb`,
  [pool](https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-oem-config/nvidia-l4t-oem-config_36.4.4-20250616085344_arm64.deb)):
  besides Ubuntu's own steps it adds four NVIDIA screens, from `nvlicense`, `nvresizefs`, `nvswap`, `nvchromium`:
  1. the **EULA**;
  2. **"APP Partition Size"**: "Please enter desired size of APP partition in Megabytes (MB). Default value in input
     field is the maximum size that can be accepted. Enter 0 or leave blank to use the maximum size value. Note that
     it is not supported to reduce the APP partition size." (This is what grows the root file system to the whole
     card: keep the default.)
  3. **"Create SWAP File (Recommended)?"**: "It is recommended to create a 4GB disk SWAP file if you intent to use the
     device for AI and Deep Learning ... may shorten life of SDCARD." Default **true**. Say yes: INT8 at 960 is the
     build most likely to run out of memory.
  4. **"Install Chromium Browser?"** default true, "Internet connection is required ... it will take several minutes."
     Say no (not needed, and it slows first boot).
  There is **no nvpmodel screen** in this package (no `nvpmodel` template ships in it), so `provision.sh`'s
  "If it asks for a power mode" will not fire. The order of the screens is not in the package: ⚠️.
- ⚠️ **Headless over the USB-C port: yes, mechanically; NVIDIA documents it only for the AGX Orin.** Package evidence,
  `/etc/systemd/nv-oem-config.sh` (36.4.4): it first runs `systemctl start nv-l4t-usb-device-mode.service`, then starts
  a throw-away X server and runs `xrandr --query | grep " connected"`. **If a display is connected it runs the GUI
  oem-config; if not, it runs the text ("debconf_ui") oem-config on the serial port `ttyGS0`**, after
  `stty -F /dev/ttyGS0 115200 cs8 -parenb -cstopb`, and logs "Please complete system configuration setup on the serial
  port provided by Jetson's USB device mode connection. e.g. /dev/ttyACMx". `/etc/nv-oem-config.conf` says
  `nv-oem-config-uart-port=ttyGS0`. The gadget (`nv-l4t-usb-device-mode-config.sh`) has USB ID **0955:7020**, "Linux for
  Tegra", with RNDIS + ECM/NCM + ACM serial + mass storage.
  So on Windows: no monitor plugged in, USB-C cable from the dev kit's USB-C port to the laptop, wait up to a minute,
  Device Manager > Ports (COM & LPT) shows **"USB Serial Device (COMn)"** (VID 0955, PID 7020), PuTTY or MobaXterm
  serial at **115200**. Windows ships the serial driver (`usbser`) in the box; NVIDIA's AGX Orin guide shows exactly
  this with no driver step ([AGX Orin Quick Start](https://docs.nvidia.com/jetson/agx-orin-devkit/user-guide/quick_start.html):
  "Make sure you do NOT connect anything into DisplayPort output", "Baud rate: 115200 for all terminal tools",
  "USB Serial Device (example: COM10)", "this CUI-based 'Jetson Initial configuration' (oem-config)"). The same gadget
  and the same script are on the Orin Nano image, but no NVIDIA page walks the Orin Nano through it, and forum posts
  show a JetPack 6 Orin Nano that sat at "Please complete system configuration setup on desktop" (forum, last resort:
  [t/285936](https://forums.developer.nvidia.com/t/jetson-orin-nano-stuck-at-first-time-boot/285936)). **Use the monitor
  as the primary route and the serial console as the fallback; do not leave a monitor plugged in while trying serial.**
  A monitor plugged in during first boot also means serial will show nothing.

### 4. Networking  ✅ Wi-Fi / ✅ 192.168.55.1 / ⚠️ Windows driver / ⚠️ sshd

- ⚠️ **openssh-server.** No NVIDIA page states it. NVIDIA's own getting-started tutorial tells the student to
  `ssh USERNAME@192.168.55.1` straight after the first-run setup with no install step
  ([Jetson AI Lab, headless setup](https://www.jetson-ai-lab.com/llms-full.txt), "Physical Setup" and "Network
  Connection"), which implies sshd is on. Keep `provision.sh`'s `systemctl is-active ssh` check and the
  `sudo apt install -y openssh-server` fallback (needs internet).
- ✅ **USB device-mode networking at 192.168.55.1 is on JetPack 6.** Package evidence (`nvidia-l4t-init` 36.4.4,
  `nv-l4t-usb-device-mode-config.sh`): `net_ip=192.168.55.1`, mask 255.255.255.0, a DHCP server that hands the laptop
  `192.168.55.100` (lease 15 s), `enable_rndis=1 enable_ecm=1 enable_acm=1 enable_ums=1`; the unit is enabled in
  `multi-user.target`. NVIDIA's [Orin Nano Hardware Layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html):
  "USB Device Mode exposes ... USB Mass Storage device for the L4T-README drive; USB Serial device ...; USB Ethernet
  (RNDIS) device ... with Jetson at 192.168.55.1". The USB-C port "does not output a display signal".
- ⚠️ **Windows 11 and no driver.** Microsoft ships the RNDIS miniport `Rndismp.sys` in every supported Windows
  ([Microsoft Learn, RNDIS](https://learn.microsoft.com/en-us/windows-hardware/drivers/network/remote-ndis--rndis-2)),
  and Windows shows the Jetson as a "L4T-README" drive plus a network adapter in NVIDIA's own Windows Device Manager
  screenshot, and NVIDIA tells Windows users to run `ssh USERNAME@192.168.55.1` in PowerShell. I could not test it.
  Plan for it to take a minute to come up and for the adapter to show "Unidentified network" (harmless). The
  laptop gets 192.168.55.100 from the board, so the hotspot's 192.168.137.x is unaffected.
- ✅ **Wi-Fi on the Super dev kit.** The carrier's M.2 Key-E 2230 slot is "Populated with the included wireless module"
  ([hardware layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html)); the box
  contains "802.11 plug-in WLAN & BT module preinstalled with antenna" (Getting Started Guide), and oem-config offers
  "Connect to Wireless network".
- ✅ **nmcli from the command line** (NVIDIA's own tutorial, over SSH): `sudo nmcli device wifi connect WIFI_SSID password WIFI_PASSWORD`
  then `hostname -I` for the address. For an SSID with spaces, quote it. If a hotspot is 5 GHz-only and the board does
  not see it, set the hotspot band to 2.4 GHz (the draft already says so).

---

## B. Power mode

### 5. nvpmodel IDs, default, reboot, `-q`, `jetson_clocks`  ✅ (one ⚠️)

- ✅ **Mode IDs on the Orin Nano Super 8 GB (P3767-0003/0005, the dev kit).** Read out of
  `nvidia-l4t-nvpmodel_36.4.4-20250616085344_arm64.deb`, file `/etc/nvpmodel/nvpmodel_p3767_0003_super.conf`
  ([package](https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-nvpmodel/nvidia-l4t-nvpmodel_36.4.4-20250616085344_arm64.deb)):

  ```
  < POWER_MODEL ID=0 NAME=15W >
  < POWER_MODEL ID=1 NAME=25W >
  < POWER_MODEL ID=2 NAME=MAXN_SUPER >
  < POWER_MODEL ID=3 NAME=7W >
  < PM_CONFIG DEFAULT=1 >
  ```

  Default **1 = 25W**; **MAXN SUPER is `sudo nvpmodel -m 2`** (NVIDIA's [JetPack 6.2 blog](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)
  says the same, and the [Jetson Linux 36.4.4 power page](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
  lists 15W/25W/MAXN_SUPER as IDs 0/1/2 for the 8 GB Super, "default mode is 25W (mode ID 1)", and "MAXN_SUPER mode is
  available only when flashing with the jetson-orin-nano-devkit-super configurations"). The 4 GB Super differs:
  `10W/25W/MAXN_SUPER/7W_AI/7W_CPU`, IDs 0-4. Which conf the board uses is chosen at boot by `nvpower.sh` from the
  device-tree `compatible` string (`p3767-0003-super` or `p3767-0005[-super]` map to the 8 GB files). The non-Super
  8 GB conf has only `15W`, `7W`: on a board that booted a non-Super device tree there is **no MAXN_SUPER** and
  `highest_mode()` would quietly pick 15W (see Corrections).
- ✅ **Reboot.** The binary prints "NVPM WARN: Reboot required for changing to this power mode: %d" and "DO YOU WANT TO
  REBOOT NOW? enter YES/yes to confirm:" only when the GPU `TPC_PG_MASK` changes. In the conf the mask is **240 for
  15W, 25W and MAXN_SUPER** and **252 for 7W**. So 25W -> MAXN_SUPER **does not ask to reboot**; only a switch to or from
  7W does. (The docs' wording: a reboot is needed "when a mode change requires a different GPU tpc_pg_mask value".)
  The wizard's "If nvpmodel asks to reboot, answer YES" is harmless but will not trigger.
- ⚠️ **`nvpmodel -q` without sudo.** NVIDIA's docs always write `sudo /usr/sbin/nvpmodel -q`. The binary contains no
  `geteuid`/`getuid` call and no root check; `-q` only reads `/var/lib/nvpmodel/status` (and prints
  `NV Power Mode: %s\n%d\n`). So it works for any user if that file is world-readable; I cannot see its mode without
  the board. The probe already falls back to reading the same file (`pmode:0002`), and the wizard runs `bs()` after,
  so nothing breaks either way. `nvpmodel` lives in `/usr/sbin`: Ubuntu puts `sbin` on every user's PATH.
- ✅ **Real `nvpmodel -q` output.** The format string in NVIDIA's binary is `"NV Power Mode: %s\n%d\n"`, i.e.

  ```
  NV Power Mode: MAXN_SUPER
  2
  ```

  A forum post on an Orin Nano Super shows the same two-line shape (`NV Power Mode: MAXN` / `2`; forum, last resort:
  [t/318719](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-dev-kit-performance/318719)). The status file
  is written as `pmode:%.4d` -> `pmode:0002`.
- ✅ **`jetson_clocks` does not survive a reboot.** NVIDIA: "The jetson_clocks.sh script maximizes a Jetson Orin
  performance by setting the static maximum frequencies of the CPU, GPU and EMC clocks." It also says "After running
  `jetson_clocks`, you cannot change the nvpmodel power mode." **So the order must stay `nvpmodel -m 2` first, then
  `jetson_clocks`** (the wizard does). No unit in `nvidia-l4t-init` or `nvidia-l4t-nvpmodel` 36.4.4 calls
  `jetson_clocks` at boot (`nvpower.service`, `nvpmodel.service` only run `nvpmodel -f /etc/nvpmodel.conf`, which
  restores the *mode* from the status file, not the pinned clocks). The docs do not state persistence in words: ⚠️
  on that sentence only; the wizard's power check re-runs after every boot anyway.
- ✅ **`jetson_probe.py`.** Ran `parse_nvpmodel_conf` + `highest_mode` on the real 36.4.4 files:
  8 GB Super -> `[(0,'15W'),(1,'25W'),(2,'MAXN_SUPER'),(3,'7W')]`, default 1, **highest `(2,'MAXN_SUPER')`**;
  4 GB Super -> highest `(2,'MAXN_SUPER')`; non-Super 8 GB -> `(0,'15W')`. `current_mode('NV Power Mode: MAXN_SUPER\n2')`
  -> `(2,'MAXN_SUPER')`; `'NV Power Mode: 25W\n1'` -> `(1,'25W')`. Both regexes match the real text. ✅

---

## C. Install (`study/jetson/install.sh`)

### 6. torch and torchvision wheels, cuDSS  ✅

- ✅ `curl -sIL` both: `302` to GitHub's release storage, then `200`, torch **233,143,716 bytes**, torchvision
  **1,640,289 bytes**. The torch wheel's METADATA (read by range request) is `torch 2.10.0`, `Requires-Python >=3.10`,
  `Requires-Dist: filelock, typing-extensions>=4.10.0, sympy>=1.13.3, networkx>=2.5.1, jinja2, fsspec>=0.8.5`:
  **no numpy requirement**. The same torch 2.10.0 cp310 file is on NVIDIA's Jetson AI Lab index
  `https://pypi.jetson-ai-lab.io/jp6/cu126/torch/` (`200`; `cu126` = JetPack 6 with CUDA 12.6, and JetPack 6.2.1 has
  CUDA 12.6, TensorRT 10.3, cuDNN 9.3 per the [6.2.1 page](https://developer.nvidia.com/embedded/jetpack-sdk-621)).
- ✅ [Ultralytics' Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) (source:
  [docs/en/guides/nvidia-jetson.md](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/nvidia-jetson.md),
  lines 207-222) says "Install torch 2.10.0 and torchvision 0.25.0 according to JP6.1" with exactly those two
  `pip install` URLs, then "Install cuDSS to fix a dependency issue with torch 2.10.0". JP6.1 and JP6.2.x share
  L4T R36.4 / CUDA 12.6 / Python 3.10, so the wheels apply to 6.2.1; the guide never names 6.2, and says it was tested on
  JP6.1 (Orin Nano Super), 6.0, 5.1.3, 4.6.1 and 7.2. cuDSS is needed (Ultralytics' words), and the command sequence in
  `install.sh` is the guide's, line for line (`wget`, `dpkg -i`, `cp ... /usr/share/keyrings/`, `apt-get update`,
  `apt-get -y install cudss`): ✅ current.

### 7. cuDSS URL and keyring  ✅

- ✅ `https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb`:
  `200`, `Content-Length: 103655858` (103.7 MB). The tarball used in the "No sudo" note,
  `.../redist/libcudss/linux-aarch64/libcudss-linux-aarch64-0.7.1.4_cuda12-archive.tar.xz`: `200`, 38,431,852 bytes.
- ✅ I read the first 550 KB of the .deb's data archive (range request): it installs
  `/var/cudss-local-tegra-repo-ubuntu2204-0.7.1/` and its apt source line is
  `deb [signed-by=/usr/share/keyrings/cudss-local-tegra-0777B0CD-keyring.gpg] file:///var/cudss-local-tegra-repo-ubuntu2204-0.7.1 /`;
  the package's `md5sums` lists `.../cudss-local-tegra-0777B0CD-keyring.gpg`, and its `postinst` prints
  `sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-local-tegra-0777B0CD-keyring.gpg /usr/share/keyrings/`.
  The glob `cudss-*-keyring.gpg` matches that one file. The repo's `Packages` file lists a package named **`cudss`**
  (0.7.1-1, a metapackage over `cudss-cuda-12`/`-13`, `libcudss0-cuda-12` 0.7.1.4-1): `apt-get -y install cudss` is right.
  ⚠️ `install.sh` tests `dpkg -s cudss`; that is fine.

### 8. Every pip pin (PyPI JSON, cp310 manylinux aarch64)  ✅ all

| Pin | cp310 aarch64 file on PyPI | numpy it needs |
|---|---|---|
| numpy 1.26.4 | `numpy-1.26.4-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | (is numpy) |
| ultralytics 8.4.149 | `py3-none-any` (uploaded 2026-09-12) | `numpy>=1.23.0` (non-macOS) |
| pyyaml 6.0.3 | `pyyaml-6.0.3-cp310-cp310-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | none |
| onnx 1.23.2 | `onnx-1.23.2-cp310-cp310-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl` | `numpy>=1.23.2` |
| onnxslim 0.1.98 | `py3-none-any` | none (but `ml-dtypes`, below) |
| opencv-python 4.11.0.86 | `opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | `numpy>=1.21.2` on 3.10 |
| onnxruntime 1.23.2 | `onnxruntime-1.23.2-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` | `numpy>=1.21.6` |

- ✅ **onnxruntime: pin `1.23.2`.** It is the **newest release with a cp310 aarch64 wheel**: 1.24.1 lists
  cp311-cp314 only, and every release after it (1.24.2 ... 1.31.1) has no cp310 aarch64 file; `requires_dist` for
  1.23.2 is `coloredlogs, flatbuffers, numpy>=1.21.6, packaging, protobuf, sympy`. glibc: the wheels need glibc 2.27/2.28;
  Ubuntu 22.04 has 2.35. The placeholder was right.
- ✅ **Nothing requires numpy>=2** at these pins. The one trap is `onnxslim` -> `ml-dtypes`: the newest `ml-dtypes`
  0.6.0 needs `numpy>=2.0.0`, but pip backtracks. I ran pip's resolver for the aarch64/cp310 target
  (`pip install --dry-run --platform manylinux_2_28_aarch64 --python-version 3.10 --only-binary=:all:`) on
  `pyyaml onnx onnxslim opencv-python onnxruntime numpy==1.26.4`: it resolves cleanly, picking **ml-dtypes 0.5.4**,
  protobuf 7.36.2, flatbuffers 25.12.19, sympy 1.14.0. (I could not resolve `ultralytics` itself that way: its
  `ultralytics-platform` dependency is marked `python_version >= "3.11"` and the host Python is 3.12; on the board's
  3.10 the marker is false. Ultralytics 8.4.149's other dependencies: cloudpickle, filelock, matplotlib, pillow,
  requests, torch, torchvision, psutil, polars, nvidia-ml-py, ultralytics-thop; no scipy.)
- ✅ The pin list keeps `numpy==1.26.4` last, so pip never upgrades it.

### 9. TensorRT's Python binding on the 6.2.1 image  ⚠️ -> fix in install.sh

- ✅ NVIDIA's Jetson AI Lab says "If you used an NVIDIA-supplied SD card image ... all necessary JetPack components
  (including nvidia-containers) and Docker are already pre-installed" ([llms-full.txt](https://www.jetson-ai-lab.com/llms-full.txt)).
  But what does "JetPack components" include? From the apt index
  ([common/dists/r36.4 Packages.gz](https://repo.download.nvidia.com/jetson/common/dists/r36.4/main/binary-arm64/Packages.gz)):
  `nvidia-jetpack-runtime 6.2.1+b38` depends on `nvidia-tensorrt 6.2.1+b38`, which depends only on **`tensorrt-libs
  10.3.0.30-1+cuda12.5`** (C++ libs, no Python). `python3-libnvinfer 10.3.0.30-1+cuda12.5`
  (`pool/main/t/tensorrt/python3-libnvinfer_10.3.0.30-1+cuda12.5_arm64.deb`) is pulled in only by
  `nvidia-jetpack-dev` -> `nvidia-tensorrt-dev` -> `tensorrt` -> `python3-libnvinfer-dev` -> `python3-libnvinfer`.
  Whether the SD image carries the `-dev` set is not stated on any NVIDIA page. **So `install.sh` must not assume it.**
  Its `check()` would print `tensorrt=MISSING(ModuleNotFoundError...)` and fail the install at the end, after 3 GB of
  downloads.
- ✅ **Does a `--system-site-packages` venv import it?** The package installs
  `/usr/lib/python3.10/dist-packages/tensorrt/{__init__.py,tensorrt.so}` and `tensorrt-10.3.0.dist-info`
  (`__version__ = "10.3.0"`, which `check()` shortens to `10.3`). Debian's `site.py` puts `dist-packages` on the path
  of the system Python, and a venv with `include-system-site-packages = true` inherits the system site dirs, so
  `import tensorrt` works from `~/uav-venv`. This is the standard Jetson arrangement; I did not run it (no board).
- Fix (apt step before the venv): see Corrections #1. Install only the Python binding: `sudo apt-get install -y python3-libnvinfer`
  (its dependencies `libnvinfer10`, `libnvinfer-plugin10`, `libnvinfer-vc-plugin10`, `libnvonnxparsers10` are the
  runtime libs the image already has, all pinned `= 10.3.0.30-1+cuda12.5`).
  The image's apt source is NVIDIA's `r36.4` repo, so `apt update` needs internet. Do **not** run `apt upgrade`: that
  repo now has L4T 36.4.7 (JetPack 6.2.2) and would move the board off 6.2.1.

### 10. Ultralytics 8.4.149 TensorRT export on a Jetson, with `YOLO_AUTOINSTALL=false`  ✅

Read at tag v8.4.149: [exporter.py](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.149/ultralytics/engine/exporter.py),
[utils/export/engine.py](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.149/ultralytics/utils/export/engine.py),
[utils/checks.py](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.149/ultralytics/utils/checks.py),
[cfg/default.yaml](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.149/ultralytics/cfg/default.yaml).

- ✅ **What must be installed beforehand.** `check_requirements()` with `AUTOINSTALL` false and a missing package just
  **returns False** (checks.py L602-L626); it does not raise. The export then fails later with an ImportError. Needed:
  - `export_onnx` (L1064-1075): `onnx>=1.12.0,<2.0.0`; because `simplify=True` is the default
    (default.yaml L90), also an onnxruntime build: with `device=0` it asks for `onnxruntime-gpu`, but it is a tuple of
    interchangeable candidates `("onnxruntime-gpu", "onnxruntime", "onnxruntime-gpu", "onnxruntime-qnn")`, so the CPU
    `onnxruntime` satisfies it; and `onnxslim>=0.1.82`. The study's pins (onnx 1.23.2, onnxslim 0.1.98,
    onnxruntime 1.23.2) satisfy all three. (`onnxruntime` is therefore needed by the *export*, not only the drift
    reference.)
  - `export_engine` -> `onnx2engine` (engine.py L236-249): `import tensorrt` must work. If it does not, the code calls
    `check_tensorrt()` which would `pip install tensorrt-cu12` (a PyPI x86/aarch64 build that does not fit JetPack),
    and with AUTOINSTALL off it fails. So the apt `python3-libnvinfer` is mandatory (item 9). TensorRT 10.2.0 is
    explicitly rejected; 10.3.0 is fine. `onnx` is imported at the top.
  - No Jetson-specific numpy pin appears in the export path. The only numpy constraints in the 8.4.149 metadata are the
    macOS and extras ones (item 8).
- ✅ **Arguments valid at 8.4.149.** The export table row for TensorRT (exporter.py L181-188) lists
  `["batch","data","dynamic","quantize","opset","simplify","workspace","nms","fraction"]`; `device`, `imgsz` and
  `format` are base arguments. `quantize=16` (FP16) and `quantize=8` (INT8) are both accepted for `engine`
  (`FP16_FORMATS` and `INT8_FORMATS` contain `"engine"`); `data="VisDrone.yaml"` is needed for INT8
  (`quantize=8` with no `data` silently takes the default dataset, L839-844); `workspace` is a float in GiB, applied as
  `int(workspace*(1<<30))` to `set_memory_pool_limit(WORKSPACE, ...)` (and ignored if `None`: TensorRT decides);
  `device=0` is also what the exporter sets by itself for `engine`; `batch=1` is valid. All of
  `studyexport.py`'s `tensorrt-fp16` / `tensorrt-int8` variants are valid.
- ✅ **INT8 calibration.** `get_int8_calibration_dataloader` (L1008-1048): `split = self.args.split or "val"` (default
  `val`), `fraction=1.0` -> **all val images**, `build_dataloader(batch=args.batch, drop_last=True)`. With `batch=1`
  that is **all 548 VisDrone val images**, at the export `imgsz`; with batch 16 `drop_last` would drop the last 4 (544).
  A warning fires below 300 images. The calibrator is **MINMAX** (`MINMAX_CALIBRATION`; `ENTROPY_CALIBRATION_2` only
  for DLA), the head Sigmoid layers are kept in FP32 (`OBEY_PRECISION_CONSTRAINTS`), and the cache is written beside
  the ONNX as `<name>.cache` and **reused on a rerun** (`read_calibration_cache`), so delete it to recalibrate.
  This matches `studyexport.py`'s header and manifest fields.
- ✅ **YOLO26 and INT8 on JetPack 6** (exporter.py L693-715): with `quantize==8`, TensorRT `>=10.3.0,<10.4.0` and
  `is_jetson(jetpack=6)` (reads `/etc/nv_tegra_release` for "R36"), the exporter sets `model.end2end = False`
  ("TensorRT 10.3.0 on JetPack 6 with INT8 cannot build NMS-free exports; selecting one-to-many outputs", issue
  ultralytics#23841). So **YOLO26s INT8 engines have the one-to-many head** (raw boxes, needs NMS) while YOLO26s FP16
  keeps the one-to-one head. `jetson_probe.py`'s `head_row()` already says this. The runner must decode each engine
  according to its header's `end2end` (`engine_header` returns it). Ultralytics' fix is TensorRT >= 10.7, which
  JetPack 6 does not have.

---

## D. Runner

### 11. The `TensorRT` class in `studybench.py`  ✅

I extracted `tensorrt.so` from the actual `python3-libnvinfer_10.3.0.30-1+cuda12.5` package (NVIDIA apt repo) and checked
that every name the class uses is present in its binary (and `nptype` / `__version__ = "10.3.0"` in its `__init__.py`):

`trt.Runtime`, `trt.Logger`, `deserialize_cuda_engine`, `create_execution_context`, `num_io_tensors`, `get_tensor_name`,
`get_tensor_mode`, `trt.TensorIOMode.INPUT`, `get_tensor_shape`, `get_tensor_dtype`, `trt.nptype`, `set_tensor_address`,
`execute_async_v3`, `device_memory_size_v2` (guarded by `getattr`, with `device_memory_size` as fallback: both exist),
`create_engine_inspector`, `get_engine_information`, `trt.LayerInformationFormat.JSON`. ✅ All valid on TensorRT 10.3.

- ✅ **The CUDA binding is PyTorch**, not pycuda/cuda-python: `torch.cuda.Stream()`, `stream.cuda_stream` (the handle
  passed to `execute_async_v3`), `torch.empty(..., pin_memory=True)`, `device.copy_(host, non_blocking=True)`,
  `torch.cuda.get_device_name(0)`. All exist in torch 2.10 and are available after the install (the Jetson wheel is
  CUDA-enabled; `check()` requires `cuda=True`). JetPack ships no Python CUDA binding, as the class's comment says.
- ✅ The engine file layout matches the exporter: Ultralytics writes `len(meta).to_bytes(4,"little",signed=True)` +
  JSON + the engine (engine.py L449-455), and `engine_header()` is used to skip it.
- ⚠️ One honest limit: none of this has run on a board. The first FP16 build at 640 is the real test.

---

## E. Time

### 12. Ultralytics' published Orin Nano Super numbers, and build time  ✅ numbers / ❌ no build-time source

- ✅ **YOLO11s, Orin Nano Super, 640, TensorRT** (Ultralytics' Jetson guide as of tags v8.3.200 and v8.4.0, "Benchmarked
  with Ultralytics 8.3.157"; [docs/en/guides/nvidia-jetson.md @ v8.3.200](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.200/docs/en/guides/nvidia-jetson.md)):
  **FP16 6.93 ms/im (mAP50-95 0.5779), INT8 5.08 ms/im (0.5735)**. Inference time excludes pre- and post-processing;
  the board was on JetPack 6.1.
- ✅ **YOLO26s, same board** (current guide, "Benchmarked with Ultralytics 8.4.33"): **FP16 7.17 ms/im, INT8 5.25
  ms/im** (mAP50-95 0.5649 / 0.5468). The current guide has **YOLO26 only**; YOLO11 numbers were removed from it.
  Our `studybench.py` times the full GPU round trip at batch 1, and adds pre/post, so expect more than these.
- ❌ **Engine build time: no primary source.** Ultralytics' guide gives none ("TensorRT profiles and tunes an engine on
  its build GPU"), and NVIDIA's pages give none. The repo's "each build takes minutes" is an estimate; do not quote a
  number. Plan for INT8 (548 calibration images + tactic search) to take longest and for 960 INT8 to be the memory risk.

---

## Corrections

(Old text -> new text. Nothing was edited; these are for the next session.)

1. **`study/jetson/install.sh`: add the TensorRT Python binding** (item 9). After the block
   `if [ ${#need[@]} -gt 0 ]; then ... fi` and before `if [ ! -x "$VENV/bin/python" ]; then`, insert:
   ```bash
   if ! python3 -c "import tensorrt" 2>/dev/null; then
     echo "== apt: python3-libnvinfer (TensorRT's Python bindings; nvidia-jetpack-runtime does not include them)"
     sudo apt-get update && sudo apt-get install -y python3-libnvinfer
   fi
   ```
   (`--check` stays as is: it imports `tensorrt` from the venv.)
2. **`study/jetson/install.sh` line `ORT=1.23.2`**: keep the value; replace nothing but the comment. Old: no comment.
   New: `ORT=1.23.2   # the newest onnxruntime with a cp310 manylinux aarch64 wheel (1.24+ dropped cp310); needs numpy>=1.21.6`.
   Also in the header comment: "onnxruntime is the CPU build, for the session runner's drift reference (#8, #39)"
   -> append "; Ultralytics' ONNX step also needs one (onnxslim/simplify), so install it before any export".
3. **`study/PLATFORMS.md`, step 2 code block.** Old: `sudo nvpmodel -m <id>     # it may ask to reboot`
   -> new: `sudo nvpmodel -m <id>     # MAXN_SUPER is mode 2 here (15W=0, 25W=1 default, MAXN_SUPER=2, 7W=3); from 25W it needs no reboot`.
   And the sentence above it ("Use the mode id the probe prints on its `run mode` line") may stay. Add after the block:
   "Run `jetson_clocks` after `nvpmodel`, never before: NVIDIA says that once `jetson_clocks` has run, the power mode
   cannot be changed."
4. **`study/PLATFORMS.md`, "What the board needs" / first-boot text and `provision.sh` line 363** (card): the line
   "A microSD card of 64 GB or more (UHS-I; NVIDIA recommends 64 GB UHS-1 or larger)" is right; add the fact "the image
   writes 24.09 GB, so a 32 GB card is the floor and a 16 GB card cannot work".
5. **`study/jetson/provision.sh` ~line 582-583 (oem-config screens).** Old:
   `step "APP partition size: keep the default (the maximum)."` /
   `step "If it asks for a power mode, pick MAXN SUPER. The wizard sets it later anyway."`
   -> new (the package has no power-mode screen; it does have swap and Chromium):
   `step "APP partition size: leave it blank or 0 (the maximum). This grows the system to the whole card."` /
   `step "Create SWAP file: Yes (4 GB; the INT8 engine builds can run out of memory without it)."` /
   `step "Install Chromium: No."` /
   `note "There is no power-mode screen on this image: the default is 25W, and the wizard sets MAXN_SUPER later."`
   Also fix the screen order wording: the official list is EULA, language, keyboard, time zone, wireless network, name /
   computer name / username / password; the NVIDIA-added screens (APP partition, swap, Chromium) appear among them in
   an order I could not confirm.
6. **`study/jetson/provision.sh` ~line 480 `# VALIDATE: whether the first boot can run headless...`.** Replace the
   comment and the note with: headless works *only if no monitor is connected*; NVIDIA's `nv-oem-config.sh` runs the
   text oem-config on the USB-C serial port at 115200 (8N1), shown on Windows as "USB Serial Device (COMn)"
   (VID 0955, PID 7020). Keep the monitor route first. Add the serial route as option B (see below).
7. **`study/jetson/provision.sh` ~line 658 `# VALIDATE: ... 192.168.55.1`.** Resolved: the board does offer USB
   networking at 192.168.55.1 over the USB-C cable once the OS is up (DHCP gives the laptop 192.168.55.100). Add it as a
   fallback candidate host: `CANDS+=("192.168.55.1")`, and in the "no SSH yet" text tell the student to plug the USB-C
   cable in. Also state that `<name>.local` over Windows' hotspot is not guaranteed (Windows' hotspot does not
   forward mDNS); the IP from `hostname -I` is the dependable route.
8. **`study/jetson/provision.sh` ~line 346 `# VALIDATE: Raspberry Pi Imager's "Use custom" takes the .zip`**: resolved
   from the Imager source (zip accepted, libarchive); keep "Not tried with this image yet" on line 442, since it was not
   run on this 24 GB image.
9. **`study/jetson/provision.sh` ~line 596 `# VALIDATE: openssh-server...`**: leave the check; no NVIDIA page states it.
10. **`study/scripts/jetson_probe.py`, `highest_mode()` / the verdict** (new, small): on an Orin Nano whose
    `/etc/nvpmodel.conf` lists no `MAXN_SUPER` (a board booted with the non-Super device tree, conf has only 15W and
    7W) the study would run silently at 15W. Add a BLOCKER: "this is an Orin Nano Super: expected MAXN_SUPER in
    `/etc/nvpmodel.conf`; reflash with the `jetson-orin-nano-devkit-super` image". Not a tomorrow-blocker if the SD
    image is the one in item 1.
11. **`study/jetson/wizard.sh` line ~312** `note "... If nvpmodel asks to reboot, answer YES ..."`: keep (harmless); from
    25W it will not ask.
12. **`study/PLATFORMS.md` "Sources" paragraph**: add "JP6.2.1 is covered: the Ultralytics guide names JP6.1, which has
    the same CUDA 12.6 / TensorRT 10.3 / Python 3.10 as 6.2.1; the wheels are the ones on NVIDIA's
    `pypi.jetson-ai-lab.io/jp6/cu126`."
13. **Do not run `sudo apt upgrade`/`dist-upgrade` on the board**: NVIDIA's `r36.4` repo is now at L4T 36.4.7 (JetPack
    6.2.2), and the study is pinned to 6.2.1 / 36.4.4. (The `provision.sh` bridge text is fine; only `apt update` plus
    a named install is used.)

## What provision.sh must handle

Every manual step from a blank card to a working `ssh key` login, in order. "Sees" is what the student sees on the
screen; "types" is what they type.

0. **Before touching the board.** Windows Mobile Hotspot on (Settings > Network > Mobile hotspot; band 2.4 GHz if the
   board will not see it; power saving off). A microSD card **>= 32 GB, 64 GB UHS-1 recommended**, and a reader.
   Free disk space for the 11.73 GB zip (Etcher and Imager read the zip directly).
1. **Download** (no login): `curl -fL -C - -o jp62-r1-orin-nano-sd-card-image.zip https://developer.nvidia.com/downloads/embedded/L4T/r36_Release_v4.4/jp62-r1-orin-nano-sd-card-image.zip`
   (302 to `developer.download.nvidia.com`, then 11,725,610,175 bytes; resumes with `-C -`).
2. **Flash.** Etcher: Flash from file -> the zip, Select target -> the SD card (check its size), Flash!, accept the
   Windows admin prompt, wait 10-20 min for write and validation, click Cancel on "You need to format the disk", take
   the card out. (Imager fallback: Choose Device any, Choose OS > Use custom > the zip, Choose Storage > the card,
   customisation No, Write, Cancel any format prompt.)
3. **Insert and power on.** Card into the slot on the underside of the module; **monitor on the DisplayPort, USB
   keyboard and mouse**; plug in the 19 V barrel supply; the kit powers on by itself, the green LED near the USB-C port
   lights. Sees (firmware OK): NVIDIA logo, then the EULA screen within 1-3 minutes. Sees (firmware older than 36.0):
   black screen / UEFI shell / no boot after ~3 min. To read the version: press Esc repeatedly at the NVIDIA logo; the
   UEFI menu shows it near the top.
4. **If the firmware is old (35.x): the bridge.** (a) Flash `JP513-orin-nano-sd-card-image_b29.zip` (8.81 GB; no login;
   `https://developer.nvidia.com/downloads/embedded/l4t/r35_release_v5.0/jp513-orin-nano-sd-card-image.zip`) to a
   64 GB card the same way. (b) Boot it with the monitor, finish the Ubuntu first-boot setup, connect to the internet.
   (c) Types `sudo systemctl status nv-l4t-bootloader-config` (a scheduled update is shown; after a run it reads
   inactive/success). (d) `sudo reboot` and let the firmware update run, without removing power (**reboot 1**; progress
   text such as "Update Progress - 10%"). (e) Types `sudo nvbootctrl dump-slots-info` (NVIDIA's example shows
   "Current version: 35.5.0" at this point). (f) Types `sudo apt update` then
   `sudo apt install nvidia-l4t-jetson-orin-nano-qspi-updater`. (g) `sudo reboot` and let the QSPI update finish
   (**reboot 2**). (h) Power off, swap in the JetPack 6.2.1 card, boot it (back to step 3, firmware now 36.x).
5. **First-boot screens on the 6.2.1 image (monitor route).** Sees and does: EULA (accept); language (English),
   keyboard layout, time zone (Kolkata); wireless network (join the hotspot SSID with its password, or skip and use
   Ethernet); your name, computer name (e.g. `jetson`), username, password (the student must remember it: it is the
   `sudo` and first `ssh` password); **APP partition size** (leave blank or 0 = the whole card); **Create SWAP file**
   (Yes, 4 GB); **Install Chromium** (No). Then it finishes and restarts into the Ubuntu desktop; log in. The default
   power mode is 25W (no screen asks). (**Headless alternative**, only with *no* monitor plugged in: USB-C cable
   from the kit's USB-C port to the laptop; wait up to 1 minute; Device Manager > Ports > "USB Serial Device (COMn)";
   PuTTY serial, COMn, 115200; the text oem-config appears on that terminal with the same screens.)
6. **Possible extra reboot for a UEFI capsule.** On the board's terminal (Ctrl+Alt+T): types
   `sudo systemctl status nv-l4t-bootloader-config`; if an update is scheduled, `sudo reboot` once and wait, no
   power-off (NVIDIA: "reboot 3, conditional").
7. **Network on the board.** If the Wi-Fi step was skipped or failed: types
   `sudo nmcli device wifi connect "SSID" password "PASSWORD"`; or plug Ethernet in. Then `hostname -I` and note the
   first address (hotspot: `192.168.137.x`). Sees the Wi-Fi icon at the top right of the desktop.
8. **sshd.** Types `systemctl is-active ssh` (expect `active`); if not: `sudo apt install -y openssh-server` then
   `sudo systemctl enable --now ssh`.
9. **Reach the board from Git Bash.** Try, in this order: `<computer-name>.local` (Windows resolves mDNS, but the
   hotspot may not forward it), the IP from step 7, then the **USB fallback `192.168.55.1`** (USB-C cable to the laptop;
   Windows adds an RNDIS adapter; the laptop gets 192.168.55.100). First connection: accept the host key (`yes`); delete
   an old key for the name/IP if "REMOTE HOST IDENTIFICATION HAS CHANGED" (the card is new).
10. **Install the laptop's key:** `ssh-keygen -t ed25519` (no passphrase, `~/.ssh/id_ed25519`) if missing;
    `ssh-copy-id -i ~/.ssh/id_ed25519.pub USER@HOST` (types the board password once); then `ssh -o BatchMode=yes USER@HOST true`
    must succeed. Git Bash has `/usr/bin/ssh`, `ssh-copy-id`, `ssh-keygen` (OpenSSH 10.3p1 here).
11. **Record facts** (`/proc/device-tree/model`, `/etc/nv_tegra_release`, `nvidia-jetpack` version, disk size) into
    `verification/jetson_facts.txt`; stop with an error if disk < ~64 GB (the install, engines and results need room)
    or if the model does not say Orin Nano. Do **not** run `apt upgrade`.
12. Hand over to `wizard.sh`: `J_HOST` and `J_USER` in `jetson/.env`.

## Sources

NVIDIA (primary):
- JetPack 6.2.1 page: <https://developer.nvidia.com/embedded/jetpack-sdk-621>
- 6.2.1 image: <https://developer.nvidia.com/downloads/embedded/L4T/r36_Release_v4.4/jp62-r1-orin-nano-sd-card-image.zip> -> <https://developer.download.nvidia.com/embedded/L4T/r36_Release_v4.4/jetson-orin-nano-devkit-super-SD-image_JP6.2.1.zip>
- JetPack 5.1.3 page: <https://developer.nvidia.com/embedded/jetpack-sdk-513>; image <https://developer.download.nvidia.com/embedded/L4T/r35_Release_v5.0/JP513-orin-nano-sd-card-image_b29.zip>
- JetPack 6.x Update Path: <https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html>
- Getting Started Guide (JetPack 6 era; the live URL now serves the JetPack 7.2.1 guide, so read it from the Dec 2025 snapshot): <https://web.archive.org/web/20251225062626/https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit>
- Orin Nano Developer Kit hardware layout (USB device mode, Wi-Fi module): <https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html>
- JetPack 6.2 install guide: <https://docs.nvidia.com/jetson/jetpack/6.2/install-setup/index.html>
- Jetson Linux 36.4.4 power modes: <https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html>
- JetPack 6.2 Super Mode blog: <https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/>
- AGX Orin Quick Start (headless oem-config over USB-C, same L4T 36 gadget): <https://docs.nvidia.com/jetson/agx-orin-devkit/user-guide/quick_start.html>
- Jetson AI Lab (NVIDIA): <https://www.jetson-ai-lab.com/tutorials/getting-started-with-jetson/> and <https://www.jetson-ai-lab.com/llms-full.txt> (headless USB setup, `nmcli`, 192.168.55.1)
- NVIDIA apt repo, L4T 36.4.4 packages read for this note: `nvidia-l4t-nvpmodel` <https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-nvpmodel/nvidia-l4t-nvpmodel_36.4.4-20250616085344_arm64.deb>, `nvidia-l4t-oem-config` <https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-oem-config/nvidia-l4t-oem-config_36.4.4-20250616085344_arm64.deb>, `nvidia-l4t-init` <https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-init/nvidia-l4t-init_36.4.4-20250616085344_arm64.deb>, package index <https://repo.download.nvidia.com/jetson/common/dists/r36.4/main/binary-arm64/Packages.gz> (and `t234/dists/r36.4/...`)
- cuDSS: <https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb>
- Jetson AI Lab wheel index: <https://pypi.jetson-ai-lab.io/jp6/cu126/torch/>

Ultralytics:
- Source at v8.4.149: <https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.149/ultralytics/engine/exporter.py>, <.../ultralytics/utils/export/engine.py>, <.../ultralytics/utils/checks.py>, <.../ultralytics/cfg/default.yaml>
- NVIDIA Jetson guide: <https://docs.ultralytics.com/guides/nvidia-jetson/>; source at main <https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/nvidia-jetson.md>; YOLO11s numbers at <https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.200/docs/en/guides/nvidia-jetson.md>
- Wheels: <https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl>, <.../torchvision-0.25.0-cp310-cp310-linux_aarch64.whl>

Other primary:
- PyPI JSON: <https://pypi.org/pypi/{numpy/1.26.4,ultralytics/8.4.149,pyyaml/6.0.3,onnx/1.23.2,onnxslim/0.1.98,opencv-python/4.11.0.86,onnxruntime/1.23.2,onnxruntime/1.24.1,onnxruntime,ml-dtypes}/json>
- Raspberry Pi Imager source: <https://github.com/raspberrypi/rpi-imager> (`src/downloadextractthread.cpp`, `src/imagewriter.cpp`)
- Microsoft Learn, RNDIS: <https://learn.microsoft.com/en-us/windows-hardware/drivers/network/remote-ndis--rndis-2>

Forum (last resort, marked where used):
- <https://forums.developer.nvidia.com/t/jetson-orin-nano-stuck-at-first-time-boot/285936> (first-boot black screen without a monitor)
- <https://forums.developer.nvidia.com/t/jetson-orin-nano-super-dev-kit-performance/318719> (`nvpmodel -q` output shape)

Not reached / not done: no board was available, so nothing here ran on a Jetson. I could not open
`docs.raspberrypi.com` (DNS failure), so the Imager answer rests on its source code. NVIDIA states no engine build time.
