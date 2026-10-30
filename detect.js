// Pure detection logic: preprocessing, decoding and non-maximum suppression. No DOM, so it can be tested.
const SIZE = 640;

// Resize with padding (letterbox) so the picture keeps its shape, and return the scale info needed to map boxes back.
function letterboxParams(w, h) {
  const r = Math.min(SIZE / w, SIZE / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  return { r, nw, nh, dx: (SIZE - nw) / 2, dy: (SIZE - nh) / 2 };
}

// RGBA pixels (SIZE x SIZE) -> float32 CHW tensor data in 0..1
function toTensorData(rgba) {
  const n = SIZE * SIZE, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = rgba[i * 4] / 255;
    out[n + i] = rgba[i * 4 + 1] / 255;
    out[2 * n + i] = rgba[i * 4 + 2] / 255;
  }
  return out;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-9);
}

// Keep the best box and drop overlapping boxes of the same class.
function nms(boxes, thr = 0.45) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) if (keep.every(k => k.cls !== b.cls || iou(k, b) < thr)) keep.push(b);
  return keep;
}

// YOLOv8 output is [1, 84, 8400]: 4 box numbers (cx, cy, w, h) then 80 class scores, for 8400 candidates.
function decode(output, lb, minScore, srcW, srcH) {
  const N = 8400, boxes = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = -1;
    for (let c = 0; c < 80; c++) {
      const s = output[(4 + c) * N + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < minScore) continue;
    const cx = output[i], cy = output[N + i], w = output[2 * N + i], h = output[3 * N + i];
    let x = (cx - w / 2 - lb.dx) / lb.r, y = (cy - h / 2 - lb.dy) / lb.r, bw = w / lb.r, bh = h / lb.r;
    x = Math.max(0, x); y = Math.max(0, y);
    bw = Math.min(bw, srcW - x); bh = Math.min(bh, srcH - y);
    boxes.push({ x, y, w: bw, h: bh, score: best, cls });
  }
  return nms(boxes);
}

if (typeof module !== "undefined") module.exports = { SIZE, letterboxParams, toTensorData, iou, nms, decode };
