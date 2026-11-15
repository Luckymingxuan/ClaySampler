"""Compare segmented synthesis with a baseline across filter and pitch boundaries."""

import argparse
import math
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

from check_tail import midi


def check(options, root):
    rate = 44100
    with wave.open(str(root / "sample.wav"), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(b"".join(struct.pack("<h", round(3000 * math.sin(i * .04))) for i in range(rate)))
    (root / "voice.sfz").write_text(
        "<region> sample=sample.wav key=60 loop_mode=loop_continuous loop_start=0 loop_end=44099 "
        "amp_veltrack=0 ampeg_release=1.7 cutoff=1800 resonance=3 "
        "cutoff_oncc1=3000 pitch_oncc1=37\n", encoding="ascii")
    # A held note and pedal span multiple segments. CC and bend changes split the voice.
    midi(root / "long.mid", bytes.fromhex(
        "00 b0 40 7f 00 90 3c 64 87 40 b0 01 40 87 40 e0 00 50 "
        "87 40 80 3c 00 87 40 b0 40 00 00 ff 2f 00"))
    midi(root / "short.mid", bytes.fromhex("00 90 3c 64 60 80 3c 00 00 ff 2f 00"))
    for tail in ("natural", "orig"):
        outputs = []
        for name, executable, cap in (("baseline", options.baseline, 32), ("segmented", options.clay, 32),
                                      ("single", options.clay, 1)):
            output = root / f"{name}-{tail}"
            manifest = root / f"{name}-{tail}.tsv"
            manifest.write_text("".join(
                f"-\tvoice.sfz\t{source}.mid\t{output.name}/{i}.wav\n"
                for i, source in enumerate(("short", "long", "short", "long"))), encoding="utf-8")
            command = [executable.resolve(), "--jobs", manifest, "--tail-policy", tail,
                       "--gpu-batch-jobs", str(cap), "--quality-gate", "off"]
            for key in ("nvrtc", "sndfile"):
                value = getattr(options, key)
                if value:
                    command += [f"--{key}", value.resolve()]
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
            if options.require_segments and name != "baseline":
                assert "SEGMENT\t" in result.stderr, "The test did not exercise multiple segments"
            outputs.append([(output / f"{i}.wav").read_bytes() for i in range(4)])
        assert outputs[0] == outputs[1] == outputs[2], f"Segment boundary mismatch in {tail}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clay", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--sndfile", type=Path)
    parser.add_argument("--require-segments", action="store_true")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="segments-", dir=Path.cwd()) as directory:
        check(options, Path(directory))
    print("Segment boundary checks passed")
