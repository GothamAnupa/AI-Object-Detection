const test = require("node:test"), assert = require("node:assert");
const { SIZE, letterboxParams, iou, nms, decode, toTensorData } = require("../detect.js");

test("letterbox keeps the shape and centres the picture", () => {
  const lb = letterboxParams(1280, 720);
  assert.equal(lb.nw, 640); assert.equal(lb.nh, 360); assert.equal(lb.dx, 0); assert.equal(lb.dy, 140);
});

test("iou of identical and separate boxes", () => {
  const a = { x: 0, y: 0, w: 10, h: 10 };
  assert.ok(Math.abs(iou(a, a) - 1) < 1e-6);
  assert.equal(iou(a, { x: 20, y: 20, w: 5, h: 5 }), 0);
});

test("nms drops overlapping boxes of the same class only", () => {
  const boxes = [
    { x: 0, y: 0, w: 10, h: 10, score: 0.9, cls: 0 },
    { x: 1, y: 1, w: 10, h: 10, score: 0.8, cls: 0 },
    { x: 1, y: 1, w: 10, h: 10, score: 0.7, cls: 5 },
  ];
  const k = nms(boxes);
  assert.equal(k.length, 2); assert.deepEqual(k.map(b => b.cls).sort(), [0, 5]);
});

test("decode maps a box back to the original picture", () => {
  const N = 8400, out = new Float32Array(84 * N);
  // one candidate at index 3: centre (320,320), size 100x100, class 5 with score 0.9
  out[0 * N + 3] = 320; out[1 * N + 3] = 320; out[2 * N + 3] = 100; out[3 * N + 3] = 100; out[(4 + 5) * N + 3] = 0.9;
  const lb = letterboxParams(1280, 1280);              // scale 0.5, no padding
  const found = decode(out, lb, 0.5, 1280, 1280);
  assert.equal(found.length, 1); assert.equal(found[0].cls, 5);
  assert.ok(Math.abs(found[0].x - 540) < 1e-3 && Math.abs(found[0].w - 200) < 1e-3);
});

test("low scores are ignored and tensor layout is channel-first", () => {
  const N = 8400, out = new Float32Array(84 * N); out[4 * N] = 0.2;
  assert.equal(decode(out, letterboxParams(640, 640), 0.5, 640, 640).length, 0);
  const rgba = new Uint8ClampedArray(SIZE * SIZE * 4); rgba[0] = 255; rgba[1] = 0; rgba[2] = 51;
  const t = toTensorData(rgba);
  assert.equal(t[0], 1); assert.equal(t[SIZE * SIZE], 0); assert.ok(Math.abs(t[2 * SIZE * SIZE] - 0.2) < 1e-6);
});
