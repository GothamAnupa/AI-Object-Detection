(() => {
  const $ = id => document.getElementById(id);
  const video = $("video"), photo = $("photo"), overlay = $("overlay"), ctx = overlay.getContext("2d");
  const work = document.createElement("canvas"); work.width = work.height = SIZE;
  const wctx = work.getContext("2d", { willReadFrequently: true });

  let session = null, stream = null, mode = "none", timer = null, busy = false;
  const COLORS = LABELS.map((_, i) => `hsl(${(i * 47) % 360} 85% 55%)`);

  function status(a, b = "") { $("status").textContent = a; $("substatus").textContent = b; }
  function showPlaceholder(on) { $("placeholder").style.display = on ? "flex" : "none"; }

  async function loadModel() {
    try {
      ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create("model/yolov8n.onnx", { executionProviders: ["wasm"] });
      status("Ready", "Start your camera or upload a photo");
      $("camBtn").disabled = false; $("file").disabled = false;
      window.__modelReady = true;
    } catch (e) {
      status("Could not load the model", String(e.message || e));
    }
  }

  function source() { return mode === "camera" ? video : photo; }
  function srcSize() { return mode === "camera" ? [video.videoWidth, video.videoHeight] : [photo.naturalWidth, photo.naturalHeight]; }

  async function detectOnce() {
    if (!session || busy || mode === "none") return null;
    const [w, h] = srcSize(); if (!w || !h) return null;
    busy = true;
    const t0 = performance.now();
    try {
      const lb = letterboxParams(w, h);
      wctx.fillStyle = "#727272"; wctx.fillRect(0, 0, SIZE, SIZE);
      wctx.drawImage(source(), 0, 0, w, h, lb.dx, lb.dy, lb.nw, lb.nh);
      const input = new ort.Tensor("float32", toTensorData(wctx.getImageData(0, 0, SIZE, SIZE).data), [1, 3, SIZE, SIZE]);
      const out = await session.run({ [session.inputNames[0]]: input });
      const minScore = $("confRange").value / 100;
      const found = decode(out[session.outputNames[0]].data, lb, minScore, w, h);
      const ms = performance.now() - t0;
      draw(found, w, h);
      report(found, ms);
      window.__lastDetections = found.map(f => LABELS[f.cls]);
      return found;
    } finally { busy = false; }
  }

  function draw(found, w, h) {
    overlay.width = w; overlay.height = h;
    ctx.clearRect(0, 0, w, h);
    const lw = Math.max(2, w / 300), fs = Math.max(14, w / 45);
    ctx.lineWidth = lw; ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.textBaseline = "top";
    for (const f of found) {
      const col = COLORS[f.cls], label = `${LABELS[f.cls]} ${Math.round(f.score * 100)}%`;
      ctx.strokeStyle = col; ctx.strokeRect(f.x, f.y, f.w, f.h);
      const tw = ctx.measureText(label).width + 10, th = fs + 6, ty = f.y - th < 0 ? f.y : f.y - th;
      ctx.fillStyle = col; ctx.fillRect(f.x, ty, tw, th);
      ctx.fillStyle = "#fff"; ctx.fillText(label, f.x + 5, ty + 3);
    }
  }

  let lastT = 0;
  function report(found, ms) {
    const now = performance.now();
    if (lastT) $("fps").textContent = `${(1000 / (now - lastT)).toFixed(1)} fps`;
    lastT = now;
    $("ms").textContent = `${Math.round(ms)} ms per frame`;
    $("total").textContent = `${found.length} object${found.length === 1 ? "" : "s"}`;
    const c = {}; found.forEach(f => c[LABELS[f.cls]] = (c[LABELS[f.cls]] || 0) + 1);
    const ul = $("counts"); ul.innerHTML = "";
    const keys = Object.keys(c).sort((a, b) => c[b] - c[a]);
    if (!keys.length) ul.innerHTML = '<li class="empty">Nothing yet</li>';
    keys.forEach(k => { const li = document.createElement("li"); li.innerHTML = `<span>${k}</span><b>${c[k]}</b>`; ul.appendChild(li); });
  }

  function loop() {
    clearTimeout(timer);
    if (mode !== "camera") return;
    const gap = 1000 / $("fpsRange").value;
    const t = performance.now();
    const run = $("detectToggle").checked ? detectOnce() : Promise.resolve(ctx.clearRect(0, 0, overlay.width, overlay.height));
    run.finally(() => { timer = setTimeout(loop, Math.max(0, gap - (performance.now() - t))); });
  }

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
    } catch (e) {
      status("Camera not available", "Allow camera access, or upload a photo instead"); showPlaceholder(true); return;
    }
    video.srcObject = stream; await video.play();
    mode = "camera"; $("frame").classList.add("cam"); photo.hidden = true; video.hidden = false; showPlaceholder(false);
    $("camBtn").textContent = "Stop camera";
    loop();
  }
  function stopCamera() {
    clearTimeout(timer);
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; mode = "none"; video.srcObject = null; $("frame").classList.remove("cam");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    $("camBtn").textContent = "Start camera";
    status("Camera stopped", "Start it again or upload a photo"); showPlaceholder(true);
  }

  $("camBtn").onclick = () => stream ? stopCamera() : startCamera();
  $("file").onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    if (stream) stopCamera();
    photo.onload = async () => { mode = "photo"; $("frame").classList.remove("cam"); video.hidden = true; photo.hidden = false; showPlaceholder(false); lastT = 0; await detectOnce(); window.__photoDone = true; };
    photo.src = URL.createObjectURL(f);
  };
  $("confRange").oninput = () => { $("confOut").textContent = $("confRange").value + "%"; if (mode === "photo") detectOnce(); };
  $("fpsRange").oninput = () => { $("fpsOut").textContent = $("fpsRange").value; };
  $("detectToggle").onchange = () => { document.querySelector(".switch .lbl").textContent = $("detectToggle").checked ? "Detection on" : "Detection off"; if (!$("detectToggle").checked) ctx.clearRect(0, 0, overlay.width, overlay.height); else if (mode === "photo") detectOnce(); };
  $("mirror").onchange = () => $("frame").classList.toggle("mirror", $("mirror").checked);
  $("frame").classList.add("mirror");

  loadModel();
})();
