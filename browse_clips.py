#!/opt/homebrew/Caskroom/miniforge/base/envs/ml/bin/python3
"""
KWS clip browser — local HTTP server + browser UI.
Reads Parquet shards on demand (no full preload), plays clips with the
browser's native audio engine, and can inject any clip into the Thingy.

Usage:
    python browse_clips.py
    python browse_clips.py --data /path/to/data --port 7865
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq

DATA_DIR    = Path(__file__).parent / "data"
LABELS_JSON = Path(__file__).parent / "kws_tf.json"
HTTP_PORT   = 7865

# ── Index (populated at startup) ────────────────────────────────────────────
LABELS: list[str] = []
SHARD_PATHS: dict[str, list[Path]] = {}
# INDEX[split][label_name] = [(shard_idx, row_in_shard), ...]
INDEX: dict[str, dict[str, list[tuple[int, int]]]] = {}
# RG_INFO[shard_path] = [(cum_start, cum_end_exclusive, rg_idx), ...]
RG_INFO: dict[str, list[tuple[int, int, int]]] = {}

# ── Row-group LRU cache ──────────────────────────────────────────────────────
_RG_CACHE: OrderedDict = OrderedDict()   # key=(shard_path, rg_idx) → pandas DataFrame
_RG_CACHE_MAX = 8
_RG_LOCK = threading.Lock()


def _build_index(data_dir: Path, labels_json: Path):
    global LABELS, SHARD_PATHS, INDEX, RG_INFO
    LABELS = json.loads(labels_json.read_text())["labels"]
    SHARD_PATHS = {
        "test":  sorted(data_dir.glob("test-*.parquet")),
        "train": sorted(data_dir.glob("train-*.parquet")),
    }
    for split, shards in SHARD_PATHS.items():
        INDEX[split] = {lbl: [] for lbl in LABELS}
        for shard_idx, shard in enumerate(shards):
            tbl = pq.read_table(str(shard), columns=["label"])
            label_col = tbl["label"].to_pylist()
            # Build row-group boundary map
            pf = pq.ParquetFile(str(shard))
            bounds: list[tuple[int, int, int]] = []
            cum = 0
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                bounds.append((cum, cum + n, rg))
                cum += n
            RG_INFO[str(shard)] = bounds
            # Populate per-label index
            for row, lbl_int in enumerate(label_col):
                INDEX[split][LABELS[lbl_int]].append((shard_idx, row))


def _get_wav_bytes(split: str, label: str, idx: int) -> bytes | None:
    entries = INDEX[split].get(label, [])
    if idx < 0 or idx >= len(entries):
        return None
    shard_idx, row_in_shard = entries[idx]
    shard_path = str(SHARD_PATHS[split][shard_idx])
    bounds = RG_INFO[shard_path]
    rg_idx = local_row = None
    for cum_s, cum_e, rg in bounds:
        if cum_s <= row_in_shard < cum_e:
            rg_idx, local_row = rg, row_in_shard - cum_s
            break
    cache_key = (shard_path, rg_idx)
    with _RG_LOCK:
        if cache_key in _RG_CACHE:
            _RG_CACHE.move_to_end(cache_key)
            df = _RG_CACHE[cache_key]
        else:
            pf = pq.ParquetFile(shard_path)
            df = pf.read_row_group(rg_idx).to_pandas()
            _RG_CACHE[cache_key] = df
            _RG_CACHE.move_to_end(cache_key)
            if len(_RG_CACHE) > _RG_CACHE_MAX:
                _RG_CACHE.popitem(last=False)
    return df.iloc[local_row]["audio"]["bytes"]


# ── HTML ─────────────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KWS Clip Browser</title>
<style>
:root {
  --bg: #1a1a1e; --bg2: #242428; --bg3: #2e2e33;
  --border: #3a3a40; --accent: #5b8af5; --accent2: #3a6ae8;
  --text: #e0e0e6; --text2: #8888a0; --ok: #4caf50; --warn: #f5a623; --err: #e05252;
  --radius: 8px; --font: system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 14px; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
header { display: flex; align-items: center; gap: 16px; padding: 10px 20px; border-bottom: 1px solid var(--border); background: var(--bg2); flex-shrink: 0; }
header h1 { font-size: 15px; font-weight: 600; letter-spacing: .02em; }
.split-toggle { display: flex; gap: 4px; margin-left: auto; }
.split-btn { padding: 5px 16px; border-radius: 20px; border: 1px solid var(--border); background: transparent; color: var(--text2); cursor: pointer; font-size: 13px; transition: all .15s; }
.split-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.layout { display: flex; flex: 1; overflow: hidden; }
.sidebar { width: 190px; border-right: 1px solid var(--border); overflow-y: auto; flex-shrink: 0; padding: 8px 0; }
.label-item { display: flex; justify-content: space-between; align-items: center; padding: 7px 14px; cursor: pointer; border-left: 3px solid transparent; transition: all .1s; }
.label-item:hover { background: var(--bg3); }
.label-item.active { background: var(--bg3); border-left-color: var(--accent); color: var(--accent); }
.label-item .lname { font-size: 13px; }
.label-item .lcount { font-size: 11px; color: var(--text2); font-variant-numeric: tabular-nums; }
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; padding: 20px; gap: 16px; }
.clip-header { display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
.clip-label-name { font-size: 18px; font-weight: 600; }
.clip-count { color: var(--text2); font-size: 13px; }
.nav-row { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
.nav-btn { width: 34px; height: 34px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--bg2); color: var(--text); cursor: pointer; font-size: 16px; display: flex; align-items: center; justify-content: center; transition: background .1s; }
.nav-btn:hover { background: var(--bg3); }
.idx-display { font-variant-numeric: tabular-nums; color: var(--text2); min-width: 80px; text-align: center; font-size: 13px; }
input[type=range] { flex: 1; accent-color: var(--accent); }
.rand-btn { padding: 6px 14px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--bg2); color: var(--text2); cursor: pointer; font-size: 13px; transition: all .1s; }
.rand-btn:hover { background: var(--bg3); color: var(--text); }
.waveform-wrap { flex-shrink: 0; background: var(--bg2); border-radius: var(--radius); border: 1px solid var(--border); overflow: hidden; height: 80px; }
canvas { width: 100%; height: 100%; display: block; }
.player-row { flex-shrink: 0; display: flex; align-items: center; gap: 10px; }
audio { flex: 1; height: 36px; border-radius: 4px; }
.inject-panel { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; flex-shrink: 0; }
.inject-panel h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--text2); margin-bottom: 10px; }
.inject-row { display: flex; gap: 8px; align-items: center; }
.inject-row label { font-size: 12px; color: var(--text2); white-space: nowrap; }
.inject-row input { flex: 1; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); padding: 6px 10px; font-size: 13px; font-family: monospace; }
.inject-btn { padding: 6px 18px; border-radius: var(--radius); border: none; background: var(--accent); color: #fff; cursor: pointer; font-size: 13px; font-weight: 500; transition: background .1s; white-space: nowrap; }
.inject-btn:hover { background: var(--accent2); }
.inject-btn:disabled { opacity: .5; cursor: default; }
.inject-output { margin-top: 10px; font-family: monospace; font-size: 12px; white-space: pre-wrap; max-height: 120px; overflow-y: auto; color: var(--text2); display: none; }
.inject-output.visible { display: block; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.dot-ok { background: var(--ok); }
.dot-err { background: var(--err); }
.kbd { font-size: 11px; color: var(--text2); margin-left: auto; }
.kbd kbd { background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; font-family: monospace; }
</style>
</head>
<body>
<header>
  <h1>KWS Clip Browser</h1>
  <div class="split-toggle">
    <button class="split-btn active" onclick="setSplit('test')">Test</button>
    <button class="split-btn" onclick="setSplit('train')">Train</button>
  </div>
  <span class="kbd"><kbd>←</kbd><kbd>→</kbd> navigate &nbsp; <kbd>R</kbd> random &nbsp; <kbd>Space</kbd> play</span>
</header>
<div class="layout">
  <div class="sidebar" id="sidebar"></div>
  <div class="main">
    <div class="clip-header">
      <span class="clip-label-name" id="clipLabel">—</span>
      <span class="clip-count" id="clipCount"></span>
    </div>
    <div class="nav-row">
      <button class="nav-btn" onclick="navigate(-1)" title="Previous (←)">‹</button>
      <span class="idx-display" id="idxDisplay">— / —</span>
      <button class="nav-btn" onclick="navigate(1)" title="Next (→)">›</button>
      <input type="range" id="idxSlider" min="0" value="0" oninput="navigateTo(+this.value)">
      <button class="rand-btn" onclick="pickRandom()">Random</button>
    </div>
    <div class="waveform-wrap">
      <canvas id="waveCanvas"></canvas>
    </div>
    <div class="player-row">
      <audio id="audioPlayer" controls preload="auto"></audio>
    </div>
    <div class="inject-panel">
      <h3>Inject to Thingy</h3>
      <div class="inject-row">
        <label>Port</label>
        <input type="text" id="portInput" value="/dev/cu.usbmodem101" spellcheck="false">
        <button class="inject-btn" id="injectBtn" onclick="doInject()">▶ Inject</button>
      </div>
      <div class="inject-output" id="injectOutput"></div>
    </div>
  </div>
</div>
<script>
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
let state = { split: 'test', label: null, idx: 0, count: 0, labels: [], counts: {} };
let currentBlobUrl = null;

async function init() {
  const r = await fetch('/api/labels');
  const d = await r.json();
  state.labels = d.labels;
  state.counts = d.counts;
  renderSidebar();
  if (state.labels.length) selectLabel(state.labels[0]);
}

function renderSidebar() {
  const el = document.getElementById('sidebar');
  el.innerHTML = state.labels.map(lbl => {
    const cnt = (state.counts[state.split]?.[lbl] ?? 0);
    const active = lbl === state.label ? ' active' : '';
    return `<div class="label-item${active}" onclick="selectLabel('${lbl}')">
      <span class="lname">${lbl}</span><span class="lcount">${cnt}</span></div>`;
  }).join('');
}

function setSplit(s) {
  state.split = s;
  document.querySelectorAll('.split-btn').forEach((b, i) => {
    b.classList.toggle('active', (i === 0) === (s === 'test'));
  });
  renderSidebar();
  if (state.label) selectLabel(state.label);
}

function selectLabel(lbl) {
  state.label = lbl;
  state.idx = 0;
  state.count = state.counts[state.split]?.[lbl] ?? 0;
  renderSidebar();
  document.getElementById('clipLabel').textContent = lbl;
  const slider = document.getElementById('idxSlider');
  slider.max = Math.max(0, state.count - 1);
  loadClip();
}

function navigate(delta) {
  const n = state.count;
  if (!n) return;
  state.idx = ((state.idx + delta) % n + n) % n;
  loadClip();
}

function navigateTo(idx) {
  state.idx = idx;
  loadClip();
}

function pickRandom() {
  if (!state.count) return;
  state.idx = Math.floor(Math.random() * state.count);
  loadClip();
}

async function loadClip() {
  if (!state.label) return;
  const { split, label, idx, count } = state;
  document.getElementById('idxDisplay').textContent = `${idx + 1} / ${count}`;
  document.getElementById('idxSlider').value = idx;

  const url = `/api/audio?split=${split}&label=${encodeURIComponent(label)}&idx=${idx}`;
  try {
    const resp = await fetch(url);
    if (!resp.ok) return;
    const blob = await resp.blob();

    // Playback
    if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
    currentBlobUrl = URL.createObjectURL(blob);
    const player = document.getElementById('audioPlayer');
    player.src = currentBlobUrl;

    // Waveform (decode async, don't block playback)
    blob.arrayBuffer().then(buf => audioCtx.decodeAudioData(buf.slice(0))).then(decoded => {
      drawWaveform(decoded.getChannelData(0));
    }).catch(() => {});
  } catch(e) { console.error(e); }
}

function drawWaveform(samples) {
  const canvas = document.getElementById('waveCanvas');
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const step = Math.ceil(samples.length / W);
  const mid = H / 2;
  const style = getComputedStyle(document.documentElement);
  ctx.strokeStyle = style.getPropertyValue('--accent').trim();
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < W; x++) {
    let min = 1, max = -1;
    for (let s = x * step; s < (x + 1) * step && s < samples.length; s++) {
      if (samples[s] < min) min = samples[s];
      if (samples[s] > max) max = samples[s];
    }
    const y1 = mid + min * mid * 0.9;
    const y2 = mid + max * mid * 0.9;
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y2);
  }
  ctx.stroke();

  // Zero line
  ctx.strokeStyle = style.getPropertyValue('--border').trim();
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();
}

async function doInject() {
  const btn = document.getElementById('injectBtn');
  const out = document.getElementById('injectOutput');
  const port = document.getElementById('portInput').value.trim();
  if (!state.label) return;
  btn.disabled = true;
  btn.textContent = '⏳ Injecting...';
  out.className = 'inject-output visible';
  out.textContent = 'Sending PCM to device...';
  try {
    const r = await fetch('/api/inject', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({split: state.split, label: state.label, idx: state.idx, port})
    });
    const d = await r.json();
    if (d.error) {
      out.innerHTML = `<span class="status-dot dot-err"></span>${d.error}`;
    } else {
      out.innerHTML = `<span class="status-dot dot-ok"></span>` + escHtml(d.output);
    }
  } catch(e) {
    out.innerHTML = `<span class="status-dot dot-err"></span>Error: ${e.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Inject';
  }
}

function escHtml(s) {
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft')  { navigate(-1); e.preventDefault(); }
  if (e.key === 'ArrowRight') { navigate(1);  e.preventDefault(); }
  if (e.key === 'r' || e.key === 'R') pickRandom();
  if (e.key === ' ') { document.getElementById('audioPlayer').paused
    ? document.getElementById('audioPlayer').play()
    : document.getElementById('audioPlayer').pause(); e.preventDefault(); }
});

window.addEventListener('resize', () => {
  const player = document.getElementById('audioPlayer');
  if (!player.paused) return;
  // Redraw waveform if needed (canvas size changed)
});

init();
</script>
</body>
</html>"""

# ── HTTP handler ─────────────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request noise

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/":
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/api/labels":
            counts: dict[str, dict[str, int]] = {}
            for split in ("test", "train"):
                counts[split] = {lbl: len(INDEX[split][lbl]) for lbl in LABELS}
            self._json({"labels": LABELS, "counts": counts})

        elif parsed.path == "/api/audio":
            split = qs.get("split", ["test"])[0]
            label = qs.get("label", [LABELS[0] if LABELS else ""])[0]
            idx   = int(qs.get("idx", [0])[0])
            wav = _get_wav_bytes(split, label, idx)
            if wav is None:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", len(wav))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(wav)

        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == "/api/inject":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            result = _do_inject(body)
            self._json(result)
        else:
            self.send_response(404); self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def _do_inject(body: dict) -> dict:
    """Call inject_clip.py as a subprocess and return {output, error}."""
    split = body.get("split", "test")
    label = body.get("label", "")
    idx   = int(body.get("idx", 0))
    port  = body.get("port", "")
    if not port:
        return {"error": "No port specified"}
    python = sys.executable
    inject_script = str(Path(__file__).parent / "inject_clip.py")
    cmd = [python, inject_script, "--split", split, "--label", label,
           "--index", str(idx), "--port", port]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out = result.stdout + result.stderr
        if result.returncode != 0:
            return {"error": out or f"Exit code {result.returncode}"}
        return {"output": out}
    except subprocess.TimeoutExpired:
        return {"error": "Inject timed out (60 s)"}
    except Exception as e:
        return {"error": str(e)}


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data",   default=str(DATA_DIR),    help="Parquet data directory")
    ap.add_argument("--labels", default=str(LABELS_JSON), help="kws_tf.json with label list")
    ap.add_argument("--port",   default=HTTP_PORT, type=int, help="HTTP port (default 7865)")
    ap.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = ap.parse_args()

    print("Building clip index...", end=" ", flush=True)
    t0 = time.time()
    _build_index(Path(args.data), Path(args.labels))
    total = sum(len(v) for split in INDEX.values() for v in split.values())
    print(f"done. {total} clips indexed in {time.time()-t0:.1f}s")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    url = f"http://localhost:{args.port}"
    print(f"Serving at {url}  (Ctrl+C to quit)")

    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=[url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
