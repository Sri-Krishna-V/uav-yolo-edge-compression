"""Per-channel INT8 weights on ONNX Runtime's stock QDQ INT8 export (#41), with no new calibration.

The stock recipe (Ultralytics' onnx_int8_quantize: quantize_static with its defaults, over Conv, Gemm and MatMul) stores
every Conv weight per tensor: one scale for all of a conv's output channels. YOLO26s's Detect head has a 32-channel box
branch (YOLO11s: 64) whose channels' weight ranges differ 4-10x, so one scale rounds the small channels' weights mostly
away, and B2 lost 16.6 % AP to INT8 against B1's 6.6 % (.scratch/uav-yolo-edge/findings/41-yolo26s-int8.md).

per_channel() re-quantizes each Conv's weights per output channel from the FP32 export (symmetric INT8, scale
absmax/127, DequantizeLinear axis 0), and its int32 bias at input scale x weight scale[c], as quantize_static(
per_channel=True) stores them. Every node and every activation scale and zero point stays the stock export's, so
nothing is calibrated again. Measured on the laptop at 800: B2 -16.6 % -> -6.6 % AP, B1 unchanged.

  stats = studyqdq.per_channel("model_int8.onnx", "model.onnx", "model_int8pc.onnx")   # {"convs": n, "biases": n}

Needs numpy and onnx: the laptop builds it, and the platforms only run the result.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

QUANTIZED = "_quantized"  # quantize_static's suffix on a quantized initializer's name
SOURCE_TOLERANCE = 0.51  # quantization steps: the stock weights round the FP32 export's to within half a step


def _same_checkpoint(stock_q, stock_scale, w, name):
    """Refuse when the FP32 weights are not the ones the stock INT8 weights were rounded from."""
    if w.shape != stock_q.shape:
        raise ValueError(f"{name}: the FP32 export's weights are {list(w.shape)}, the stock INT8 export's {list(stock_q.shape)}: "
                         "not exports of the same checkpoint")
    off = float(np.abs(w.astype(np.float64) / float(stock_scale) - stock_q).max())
    if off > SOURCE_TOLERANCE:
        raise ValueError(f"{name}: the stock INT8 weights are up to {off:.2f} quantization steps from the FP32 export's: "
                         "not exports of the same checkpoint")


def per_channel(int8_path, fp32_path, out_path):
    """Write the stock INT8 export at int8_path with per-channel Conv weights, taken from the FP32 export at fp32_path,
    to out_path. Returns how many convs' weights and biases it re-quantized."""
    model = onnx.load(str(int8_path))
    g = model.graph
    floats = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(fp32_path)).graph.initializer}
    inits = {i.name: i for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    uses = {}
    for n in g.node:
        for i in n.input:
            uses[i] = uses.get(i, 0) + 1

    def dq_of(tensor):
        """The DequantizeLinear that makes tensor from a quantized initializer, else None."""
        n = producer.get(tensor)
        return n if n is not None and n.op_type == "DequantizeLinear" and n.input[0] in inits else None

    def replace(dq, *arrays):
        """Give a DequantizeLinear new initializers (its value, scale and zero point, in its input order) and axis 0."""
        for name, a in zip(dq.input, arrays):
            if uses[name] != 1:
                raise ValueError(f"{name} is shared by {uses[name]} nodes: re-quantizing it would change the others")
            inits[name].CopyFrom(numpy_helper.from_array(a, name))
        dq.attribute.append(helper.make_attribute("axis", 0))

    stats = {"convs": 0, "biases": 0}
    for n in g.node:
        wdq = dq_of(n.input[1]) if n.op_type == "Conv" else None
        if wdq is None:  # not a conv, or a conv the stock export left in float
            continue
        if any(a.name == "axis" for a in wdq.attribute) or inits[wdq.input[1]].dims:
            raise ValueError(f"{n.name}: its weights are already quantized per channel")
        base = wdq.input[0].removesuffix(QUANTIZED)
        if base not in floats:
            raise ValueError(f"{n.name}: the FP32 export has no {base}: not an export of the same checkpoint")
        _same_checkpoint(numpy_helper.to_array(inits[wdq.input[0]]), numpy_helper.to_array(inits[wdq.input[1]]), floats[base], base)

        w = floats[base].astype(np.float64)
        cout = w.shape[0]
        scale = np.maximum(np.abs(w.reshape(cout, -1)).max(1), 1e-12) / 127.0
        q = np.clip(np.round(w / scale.reshape(-1, *([1] * (w.ndim - 1)))), -127, 127).astype(np.int8)
        replace(wdq, q, scale.astype(np.float32), np.zeros(cout, np.int8))
        stats["convs"] += 1

        bdq = dq_of(n.input[2]) if len(n.input) > 2 else None
        xdq = producer.get(n.input[0])
        if bdq is None:
            continue
        if xdq is None or xdq.op_type != "DequantizeLinear":
            raise ValueError(f"{n.name}: its bias is quantized but its input is not")
        x_scale = float(numpy_helper.to_array(inits[xdq.input[1]]))
        b_scale = (x_scale * scale).astype(np.float32)
        b = floats[bdq.input[0].removesuffix(QUANTIZED)].astype(np.float64)
        bq = np.clip(np.round(b / b_scale.astype(np.float64)), -2**31, 2**31 - 1).astype(np.int32)
        replace(bdq, bq, b_scale, np.zeros(cout, np.int32))
        stats["biases"] += 1

    onnx.save(model, str(out_path))
    return stats
