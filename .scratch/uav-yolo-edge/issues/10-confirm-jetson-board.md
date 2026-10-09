# Confirm the Jetson board

Type: task
Status: open
Blocked by: none

## Question

Determine exactly which Jetson the EEE department has, and whether we control its software stack.

This single fact decides whether the Jetson is an easy target or a multi-week yak-shave, so it needs answering early even though the Jetson is benchmarked last.

**Why it matters:** the proposal deck specifies the original Jetson Nano (128-core Maxwell). That board is EOL at JetPack 4.6.x - Ubuntu 18.04, Python 3.6, CUDA 10.2, TensorRT 8.2. The ultralytics package requires Python 3.8 or newer, so YOLO11/YOLO26 cannot be pip-installed there at all. The only path would be exporting ONNX on another machine and building the engine on-device with trtexec, and the opset coverage of TensorRT 8.2 may reject modern YOLO export graphs outright. The deck own Jetson citation (ref [16]) is YOLOv7, not YOLO11 - the same constraint, visible in the literature.

If it is instead a Jetson Orin Nano, none of this applies: JetPack 6, Python 3.10, current TensorRT, and it becomes the easiest platform in the fleet.

Find out and record:

1. Exact board and module - original Jetson Nano 4GB/2GB, Xavier NX, Orin Nano, or Orin NX. The carrier board label and `cat /etc/nv_tegra_release` both settle it.
2. Current JetPack / L4T version, Python version, CUDA and TensorRT versions.
3. **Whether we can flash it ourselves** - a fresh OS from scratch, or a shared machine we must not disturb. The dev flagged this as open.
4. Physical access terms: how long we can hold it, whether it can leave the lab, and whether tegrastats power reading is available to our user account.

Resolution records the board identity, the stack versions, the flashing rights, and the resulting verdict: is the Jetson a normal platform or a special-cased one?
