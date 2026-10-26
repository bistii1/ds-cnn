#!/usr/bin/env python3
"""Drive one on-device snapshot over serial and save the full log.

Sends the `snapshot` command to the Thingy:53, records everything the firmware
prints until it sees the end marker, and writes it to a file. A human must speak
the test word into the mic while this runs (the firmware records live audio).

Usage:
    python tools/capture_snapshot.py --port /dev/cu.usbmodem101 --out before_yes.txt
"""
from __future__ import annotations

import argparse
import sys
import time

import serial  # pyserial

END_MARKER = "CAPTURE COMPLETE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/cu.usbmodem101")
    ap.add_argument("--out", required=True, help="file to write the raw log to")
    ap.add_argument("--oracle", default=None,
                    help="name of a baked-in oracle clip to select before the "
                         "snapshot (e.g. set_a_timer, volume_down, yes)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=75.0,
                    help="seconds to wait for the dump to finish (the full "
                         "PCM+MFCC dump is ~2000 lines and streams slowly)")
    args = ap.parse_args()

    with serial.Serial(args.port, args.baud, timeout=0.3) as ser:
        time.sleep(0.3)
        ser.reset_input_buffer()
        # The firmware boots with inference HALTED. The snapshot request is only
        # serviced while the ML pipeline is running (is_inferencing), so start it
        # first and give the mic/pipeline a moment to come up before snapshotting.
        ser.write(b"start\r\n")
        ser.flush()
        time.sleep(1.5)
        if args.oracle:
            ser.reset_input_buffer()
            ser.write(f"oracle {args.oracle}\r\n".encode())
            ser.flush()
            time.sleep(0.5)
        ser.reset_input_buffer()
        ser.write(b"snapshot\r\n")
        ser.flush()

        # Read raw bytes in large chunks, not line-by-line. The firmware bursts the
        # 32000-sample PCM dump fast; line-by-line reading cannot drain the OS serial
        # buffer in time and silently drops samples. Chunked reads keep up; the buffer
        # is split into lines only at the end.
        buf = bytearray()
        deadline = time.time() + args.timeout
        saw_end = False
        last_report = 0
        while time.time() < deadline:
            chunk = ser.read(8192)
            if chunk:
                buf += chunk
                # surface countdown/prompt lines live without flooding with PCM
                if time.time() - last_report > 0.5:
                    tail = buf.decode("utf-8", errors="replace").splitlines()[-1:]
                    for t in tail:
                        if "SNAPSHOT" in t or "SPEAK" in t:
                            print(t)
                    last_report = time.time()
            if END_MARKER.encode() in buf:
                saw_end = True
                break

    text = buf.decode("utf-8", errors="replace")
    with open(args.out, "w") as f:
        f.write(text)

    lines = text.splitlines()
    n_pcm = sum(1 for l in lines if l.startswith("PCM,"))
    print(f"\n[capture] wrote {args.out}  ({len(lines)} lines, {n_pcm} PCM lines)")
    if not saw_end:
        print("[capture] WARNING: never saw 'CAPTURE COMPLETE' - dump may be "
              "incomplete (device idle, wrong port, or timeout too short).")
        sys.exit(1)


if __name__ == "__main__":
    main()
