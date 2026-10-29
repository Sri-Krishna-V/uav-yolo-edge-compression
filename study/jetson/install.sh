#!/usr/bin/env bash
# PLATFORMS.md's Jetson install (step 3), on the board: JetPack 6.1 / 6.2 (L4T R36.4, Python 3.10, CUDA 12.6,
# TensorRT 10.3). jetson/wizard.sh runs it over SSH; it also runs by hand:
#
#   bash ~/uav/study/jetson/install.sh           install what is missing (sudo asks for the password once or twice)
#   bash ~/uav/study/jetson/install.sh --check   only check the install: exit 0 when every pin is in place
#
# The venv sees the system packages, because JetPack's TensorRT is an apt package (python3-libnvinfer) outside any venv.
# torch goes in before ultralytics, so pip does not fetch a PyPI torch first. onnxruntime is the CPU build, for the
# session runner's drift reference (#8, #39). Run it again at any time: a step that is done is skipped. It never runs
# apt upgrade: NVIDIA's r36.4 repository has moved on to L4T 36.4.7 (JetPack 6.2.2), and the study is on 6.2.1.
# The pins and URLs were checked on 2026-10-29 (.scratch/uav-yolo-edge/findings/39-jetson-orin-nano-super-steps.md).
set -euo pipefail

VENV="$HOME/uav-venv"
U=https://github.com/ultralytics/assets/releases/download/v0.0.0
TORCH="$U/torch-2.10.0-cp310-cp310-linux_aarch64.whl"
TORCHVISION="$U/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl"
CUDSS_DEB=cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
CUDSS_URL="https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/$CUDSS_DEB"
NUMPY=1.26.4   # NVIDIA's PyTorch-for-Jetson guide pins numpy 1.26; a precaution for JetPack's TensorRT bindings
ORT=1.23.2     # the newest onnxruntime with a cp310 manylinux aarch64 wheel (1.24 dropped cp310); it needs numpy>=1.21.6
PINS=(ultralytics==8.4.149 pyyaml==6.0.3 onnx==1.23.2 onnxslim==0.1.98 opencv-python==4.11.0.86 "onnxruntime==$ORT" "numpy==$NUMPY")
# What --check wants, as the check prints it: torch, CUDA available, torchvision, TensorRT (10.3.x), numpy, ultralytics, onnxruntime
WANT="torch=2.10.0 cuda=True torchvision=0.25.0 tensorrt=10.3 numpy=$NUMPY ultralytics=8.4.149 onnxruntime=$ORT"

check() {
  [ -x "$VENV/bin/python" ] || { echo "no venv at $VENV"; return 1; }
  local have
  have=$("$VENV/bin/python" - <<'EOF' 2>&1
import importlib
out = []
for name in ("torch", "torchvision", "tensorrt", "numpy", "ultralytics", "onnxruntime", "cv2", "onnx", "onnxslim"):
    try:
        m = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001  any import failure is a missing piece
        out.append(f"{name}=MISSING({type(e).__name__}: {e})".replace(" ", "_"))
        continue
    v = getattr(m, "__version__", "?")
    if name == "torch":
        out.append(f"torch={v.split('+')[0]} cuda={m.cuda.is_available()}")
    elif name == "tensorrt":
        out.append("tensorrt=" + ".".join(v.split(".")[:2]))
    elif name in ("cv2", "onnx", "onnxslim"):
        continue  # imported only to prove they load
    else:
        out.append(f"{name}={v.split('+')[0]}")
print(" ".join(out))
EOF
) || true
  echo "have: $have"
  echo "want: $WANT"
  local w
  for w in $WANT; do
    case " $have " in *" $w "*) ;; *) return 1 ;; esac
  done
}

if [ "${1:-}" = --check ]; then
  check
  exit
fi

if check >/dev/null 2>&1; then
  echo "the install is complete:"
  check
  exit 0
fi

need=()
python3 -c "import ensurepip, venv" 2>/dev/null || need+=(python3-venv)
python3 -m pip --version >/dev/null 2>&1 || need+=(python3-pip)
if [ ${#need[@]} -gt 0 ]; then
  echo "== apt: ${need[*]}"
  sudo apt-get update && sudo apt-get install -y "${need[@]}"
fi
# nvidia-jetpack-runtime brings TensorRT's libraries but not its Python binding, which only nvidia-jetpack-dev pulls in
if ! python3 -c "import tensorrt" 2>/dev/null; then
  echo "== apt: python3-libnvinfer (TensorRT's Python binding)"
  sudo apt-get update && sudo apt-get install -y python3-libnvinfer
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "== the venv: $VENV (with the system packages, for TensorRT)"
  python3 -m venv --system-site-packages "$VENV"
fi
PIP=("$VENV/bin/python" -m pip)
"${PIP[@]}" install -U pip
echo "== numpy $NUMPY, then the Jetson torch and torchvision wheels"
"${PIP[@]}" install "numpy==$NUMPY"
"${PIP[@]}" install "$TORCH" "$TORCHVISION"

if ! dpkg -s cudss >/dev/null 2>&1; then
  echo "== cuDSS 0.7.1, which torch 2.10 needs (Ultralytics' Jetson guide)"
  tmp=$(mktemp -d)
  wget -q --show-progress -O "$tmp/$CUDSS_DEB" "$CUDSS_URL"
  sudo dpkg -i "$tmp/$CUDSS_DEB"
  sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update && sudo apt-get -y install cudss
  rm -rf "$tmp"
fi

echo "== the study's pins: ${PINS[*]}"
"${PIP[@]}" install "${PINS[@]}"

echo "== check"
if check; then
  echo "the install is complete"
else
  echo "the install is NOT complete: compare have and want above"
  exit 1
fi
