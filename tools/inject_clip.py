"""
Stream a 1 s audio clip into the Thingy:53 over the serial console and run it
through the on-device int8 pipeline, bypassing the microphone (its level is far below the training clips).

Pair with the firmware `inject` console command. The
clip can be recorded anywhere, e.g. in Audacity; this tool resamples it to
16 kHz mono and normalizes it to a loud reference level (-20 dBFS RMS), close to
the level the training clips naturally sit at, so it reaches the model loud rather
than at the mic's level. (Training itself does not loudness-normalize; the clips
are just naturally loud.) Because the level is fixed on the host, injected clips avoid the quiet-mic
problem.

Usage:
    # real run against the board
    python tools/inject_clip.py my_word.wav --port /dev/tty.usbmodemXXXX

    # dry run: just prepare the clip and save it, no serial (for host-side checks)
    python tools/inject_clip.py my_word.wav --out-pcm prepared.npy

Requires: numpy, scipy. Real (non dry-run) also needs pyserial.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 1          # 16000, must match firmware RAW_SAMPLE_COUNT (v2, 1 s clips)
TARGET_DBFS = -20.0                       # loud reference level (near the training clips' natural level)


def load_and_prepare(path, target_dbfs, normalize=True):
    """wav (any rate/channels) -> (16000,) int16 at 16 kHz, normalized to target."""
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:                        # stereo -> mono
        x = x.mean(axis=1)
    # scipy returns int16 for PCM16 wavs; bring to [-1, 1] like soundfile/firmware
    if np.issubdtype(np.dtype(x.dtype), np.floating) is False:
        pass
    if np.max(np.abs(x)) > 1.5:           # integer PCM, scale down
        x = x / 32768.0
    if sr != SAMPLE_RATE:                 # resample to 16 kHz
        g = np.gcd(int(sr), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g)
    if normalize:
        rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        cur_dbfs = 20.0 * np.log10(rms)
        gain = 10.0 ** ((target_dbfs - cur_dbfs) / 20.0)
        x = x * gain
    # pad / trim to exactly 1 s
    if len(x) < CLIP_SAMPLES:
        x = np.concatenate([x, np.zeros(CLIP_SAMPLES - len(x))])
    else:
        x = x[:CLIP_SAMPLES]
    pcm = np.round(x * 32768.0).clip(-32768, 32767).astype(np.int16)
    return pcm


def measured_dbfs(pcm):
    f = pcm.astype(np.float64) / 32768.0
    return 20.0 * np.log10(np.sqrt(np.mean(f ** 2)) + 1e-12)


def _ack_count(ack):
    """Pull the running sample count out of '[INJECT] ok N' or '[INJECT] FULL N'."""
    m = re.search(r"(?:ok|FULL)\s+(\d+)", ack)
    return int(m.group(1)) if m else None


# Zephyr's uart_console keeps its line cursor in a uint8_t: a line longer than 255
# chars wraps and overwrites its own start, silently losing samples, regardless of
# CONFIG_CONSOLE_INPUT_MAX_LINE_LEN. Keep every line safely under that.
MAX_LINE_CHARS = 250


def pack_lines(pcm, per_line, max_chars=MAX_LINE_CHARS):
    """Split samples into comma-separated lines of <= max_chars and <= per_line samples."""
    lines, cur, cur_len = [], [], 0
    for v in pcm:
        s = str(int(v))
        add = len(s) + (1 if cur else 0)
        if cur and (cur_len + add > max_chars or len(cur) >= per_line):
            lines.append(cur)
            cur, cur_len, add = [], 0, len(s)
        cur.append(s)
        cur_len += add
    if cur:
        lines.append(cur)
    return lines


def stream_to_device(pcm, port, baud, per_line, read_timeout):
    import serial  # pyserial, imported lazily so dry runs need no install
    checksum = int(np.sum(pcm.astype(np.int64)))   # firmware sums as a long
    capacity = len(pcm)
    ser = serial.Serial(port, baud, timeout=read_timeout)
    time.sleep(0.3)
    ser.reset_input_buffer()

    def wait_for(substr, limit=10.0):
        end = time.time() + limit
        while time.time() < end:
            ln = ser.readline().decode(errors="replace").strip()
            if ln:
                if "[INJECT]" in ln or "OUT," in ln:
                    print("  <", ln)
                if substr in ln:
                    return ln
        return None

    def cancel():
        # Always leave the device OUT of inject mode. A run that never fills the
        # buffer leaves the firmware treating the next typed line as PCM data
        # (the stuck-inject bug), so we injectcancel on every exit path.
        try:
            ser.write(b"\ninjectcancel\n")
            time.sleep(0.3)
            ser.reset_input_buffer()
        except Exception:
            pass

    # Lines are packed to <= MAX_LINE_CHARS (see pack_lines). We still verify the
    # device's running count after every line, and retry the whole clip once if a
    # line is lost, so a bad upload is caught on the first wrong line.
    lines = pack_lines(pcm, per_line)

    try:
        for attempt in (1, 2):
            print(f"[host] inject attempt {attempt}: {len(lines)} lines "
                  f"(<= {MAX_LINE_CHARS} chars each) ...")
            ser.write(b"\nstop\n")          # halt inference so the console isn't starved
            time.sleep(0.4)
            ser.write(b"injectcancel\n")    # clear any stuck inject mode, then start fresh
            time.sleep(0.4)
            ser.reset_input_buffer()
            ser.write(b"inject\n")
            if not wait_for("READY", 10.0):
                cancel()
                sys.exit("[host] device did not acknowledge 'inject' (no READY). Is the v2 firmware flashed?")

            t0 = time.time()
            expected = 0
            full_line = None
            dropped = False
            for chunk in lines:
                expected = min(expected + len(chunk), capacity)
                ser.write((",".join(chunk) + "\n").encode())
                ack = wait_for("[INJECT]", 10.0)
                if ack is None:
                    print(f"[host] no ack after sample {expected}; serial stalled.")
                    dropped = True
                    break
                if "FULL" in ack:
                    full_line = ack
                    break
                count = _ack_count(ack)
                if count is None or count != expected:
                    print(f"[host] device dropped samples (sent {expected}, device has {count}).")
                    dropped = True
                    break
            dt = time.time() - t0

            if dropped:
                cancel()
                continue  # retry once

            if full_line is None:
                full_line = wait_for("checksum=", 10.0)
            if not (full_line and "checksum=" in full_line):
                print("[host] never got the FULL/checksum line.")
                cancel()
                continue

            dev_sum = int(full_line.split("checksum=")[1].split()[0])
            ok = (dev_sum == checksum)
            print(f"[host] upload done in {dt:.1f}s ({len(lines)} lines); "
                  f"checksum host={checksum} device={dev_sum} -> {'MATCH' if ok else 'MISMATCH'}")
            if not ok:
                print("[host] checksum mismatch.")
                cancel()
                continue

            # success: the clip is loaded and the ML thread runs it. Collect OUT lines.
            outs = []
            end = time.time() + 30.0
            while time.time() < end:
                ln = ser.readline().decode(errors="replace").strip()
                if not ln:
                    continue
                if ln.startswith("OUT,") and "ERROR" not in ln:
                    parts = ln.split(",")
                    outs.append((parts[1], float(parts[3])))
                if "CAPTURE COMPLETE" in ln:
                    break
            return outs

        sys.exit("[host] inject failed twice. The device is losing serial data; "
                 "check that the kws-v2 firmware with the inject fixes is flashed.")
    finally:
        cancel()
        ser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", help="audio clip to inject (any sample rate / channels)")
    ap.add_argument("--port", help="serial port (e.g. /dev/tty.usbmodemXXXX). Omit for a dry run.")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dbfs", type=float, default=TARGET_DBFS, help="target loudness (default -20)")
    ap.add_argument("--no-normalize", action="store_true", help="skip loudness normalization")
    ap.add_argument("--per-line", type=int, default=200,
                    help="max samples per serial line; each line is also capped at "
                         f"{MAX_LINE_CHARS} chars (the device console wraps past 255)")
    ap.add_argument("--read-timeout", type=float, default=1.0)
    ap.add_argument("--out-pcm", help="save the prepared int16 clip to this .npy and exit if no --port")
    args = ap.parse_args()

    pcm = load_and_prepare(args.wav, args.dbfs, normalize=not args.no_normalize)
    checksum = int(np.sum(pcm.astype(np.int64)))
    print(f"[host] prepared {len(pcm)} samples @ {SAMPLE_RATE} Hz, "
          f"{measured_dbfs(pcm):+.1f} dBFS, checksum={checksum}")

    if args.out_pcm:
        np.save(args.out_pcm, pcm)
        print(f"[host] saved prepared clip -> {args.out_pcm}")

    if not args.port:
        print("[host] no --port given: dry run only, nothing sent.")
        return

    outs = stream_to_device(pcm, args.port, args.baud, args.per_line, args.read_timeout)
    if outs:
        outs.sort(key=lambda t: t[1], reverse=True)
        print("\n[host] device top-5:")
        for label, pct in outs[:5]:
            print(f"  {label:<14} {pct:6.1f}%")


if __name__ == "__main__":
    main()
