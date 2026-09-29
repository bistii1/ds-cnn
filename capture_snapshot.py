"""
Usage:
    python capture_snapshot.py /dev/tty.usbmodem1101
    python capture_snapshot.py /dev/tty.usbmodem1101 --compare
"""
import argparse
import subprocess
import sys
import time

import serial

DONE_MARKER = b"=== CAPTURE COMPLETE ==="
OUT_FILE    = "snapshot.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--baud",    default=115200, type=int)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--tflite",  default="kws_tf_int8.tflite")
    ap.add_argument("--config",  default="config.yaml")
    ap.add_argument("--src",     default="src")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.1,
                        rtscts=False, dsrdtr=False)
    time.sleep(0.3)
    ser.reset_input_buffer()

    ser.write(b"snapshot\r\n")
    ser.flush()
    print(f"[capture] sent 'snapshot' — writing to {OUT_FILE}")

    buf = bytearray()
    last_rx = time.time()

    with open(OUT_FILE, "wb") as f:
        while True:
            chunk = ser.read(4096)
            if chunk:
                f.write(chunk)
                f.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                buf.extend(chunk)
                last_rx = time.time()
                if DONE_MARKER in buf:
                    break
            else:
                if time.time() - last_rx > 60:
                    print("\n[capture] timeout")
                    sys.exit(1)

    ser.close()
    print(f"\n[capture] done — {len(buf)} bytes saved to {OUT_FILE}")

    if args.compare:
        subprocess.run([sys.executable, "compare_snapshot.py", OUT_FILE,
                        "--tflite", args.tflite,
                        "--config", args.config,
                        "--src",    args.src])


if __name__ == "__main__":
    main()
