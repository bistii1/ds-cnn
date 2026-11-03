"""Convert a firmware snapshot log to a playable WAV file.

Parses the PCM,<values> lines from a snapshot.txt produced by the firmware
'snapshot' command, ignoring all other log lines.

Usage:
    python pcm_to_wav.py                          # snapshot.txt -> snapshot.wav
    python pcm_to_wav.py my_log.txt               # my_log.txt   -> my_log.wav
    python pcm_to_wav.py my_log.txt output.wav    # explicit output name
"""
import struct
import sys
import wave
from pathlib import Path

SAMPLE_RATE  = 16000
CHANNELS     = 1
SAMPLE_WIDTH = 2   # 16-bit PCM

input_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("snapshot.txt")
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".wav")

samples: list[int] = []
with open(input_path) as f:
    for line in f:
        line = line.strip()
        if not line.startswith("PCM,"):
            continue                          # skip all non-PCM lines
        for value in line[4:].split(","):    # strip "PCM," prefix, split values
            v = value.strip()
            if v:
                samples.append(int(v))

if not samples:
    sys.exit(f"[error] no PCM lines found in {input_path}")

print(f"Read {len(samples)} samples  ({len(samples) / SAMPLE_RATE:.2f} s)")

packed = struct.pack(f"<{len(samples)}h", *samples)
with wave.open(str(output_path), "wb") as wav:
    wav.setnchannels(CHANNELS)
    wav.setsampwidth(SAMPLE_WIDTH)
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(packed)

print(f"Wrote: {output_path}")
