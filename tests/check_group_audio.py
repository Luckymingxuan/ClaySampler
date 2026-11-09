"""Exercise CUDA quality gating, explicit groups, and host spill through the CLI."""

import argparse
import math
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

from check_tail import RATE, midi, read_wav


def pcm(path):
    return [sample for (sample,) in struct.iter_unpack("<h", read_wav(path)[2])]


def transfer(result):
    line = next(line for line in result.stdout.splitlines() if line.startswith("TRANSFER\t"))
    return tuple(map(int, line.split("\t")[2:]))


def check(options, root):
    melody = root / "melody.mid"
    midi(melody, bytes.fromhex("00 90 3c 64 83 60 80 3c 00 00 ff 2f 00"))  # 0.5 s
    for name, value in [("soft", 1600), ("loud", 6400), ("quiet", 100), ("tiny", 8)]:
        with wave.open(str(root / f"{name}.wav"), "wb") as stream:
            stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            stream.writeframes(struct.pack("<h", value) * RATE)
        (root / f"{name}.sfz").write_text(
            f"<region> sample={name}.wav key=60 loop_mode=one_shot amp_veltrack=0\n", encoding="ascii")
    (root / "invalid.mid").write_bytes(b"not MIDI")
    executable = [options.clay.resolve()]
    for key in ("nvrtc", "sndfile"):
        value = getattr(options, key)
        if value is not None:
            executable += [f"--{key}", value.resolve()]

    def render(directory, rows, *flags, expected=0):
        manifest = root / f"{directory}.tsv"
        manifest.write_text(
            "".join(f"{group}\t{preset}.sfz\t{source}.mid\t{directory}/{output}.wav\n"
                    for group, preset, source, output in rows),
            encoding="utf-8",
        )
        result = subprocess.run(executable + ["--jobs", manifest, *flags], capture_output=True, text=True)
        assert result.returncode == expected, result.stdout + result.stderr
        return result

    rows = [
        ("pair", "soft", "melody", "soft"),
        ("-", "loud", "melody", "independent"),
        ("pair", "quiet", "melody", "quiet"),
        ("pair", "loud", "melody", "loud"),
    ]
    render("off", rows)
    grouped = render("on", rows, "--group-volume", "mean", "--quality-gate", "clayengine", expected=2)
    assert "quiet_performance" in grouped.stderr and not (root / "on/quiet.wav").exists()
    assert (root / "off/independent.wav").read_bytes() == (root / "on/independent.wav").read_bytes()
    assert len([line for line in grouped.stdout.splitlines() if line.startswith("VOLUME\t")]) == 2
    soft, loud = (pcm(root / f"off/{name}.wav") for name in ("soft", "loud"))
    body = RATE // 2
    rms = [math.sqrt(sum(x * x for x in samples[:body]) / body) for samples in (soft, loud)]
    target = sum(rms) / 2
    for name, samples, level in zip(("soft", "loud"), (soft, loud), rms):
        assert pcm(root / f"on/{name}.wav") == [round(x * (target / level)) for x in samples]
    downloaded, spilled, waiting = transfer(grouped)
    assert 0 < downloaded <= sum((root / f"on/{name}.wav").stat().st_size - 44
                                  for name in ("soft", "loud", "independent"))
    assert (spilled, waiting) == (0, 0)

    quiet = [("-", "quiet", "melody", "out")]
    rejected = render("rejected", quiet, "--quality-gate", "clayengine", expected=2)
    assert "quiet_performance" in rejected.stderr and transfer(rejected) == (0, 0, 0)
    render("custom", quiet, "--quality-gate", "custom", "--qa-min-onset-median-dbfs", "-70",
           "--qa-min-onset-quartile-dbfs", "-75")
    assert (root / "custom/out.wav").read_bytes() == (root / "off/quiet.wav").read_bytes()
    short = render("too-long", rows[:1], "--quality-gate", "custom", "--qa-max-seconds", ".6", expected=2)
    assert "rendered_audio_too_long" in short.stderr and transfer(short) == (0, 0, 0)
    incomplete = render("incomplete", [("pair", "soft", "melody", "one"), ("pair", "loud", "invalid", "two")],
                        "--group-volume", "mean", expected=2)
    assert "insufficient_qualified_group_members" in incomplete.stderr and transfer(incomplete) == (0, 0, 0)
    tiny = render("tiny", [("-", "tiny", "melody", "out")], "--quality-gate", "custom",
                  "--qa-min-onset-median-dbfs", "-120", "--qa-min-onset-quartile-dbfs", "-120",
                  "--qa-max-silence-fraction", "1", expected=2)
    assert "effectively_silent_audio" in tiny.stderr and transfer(tiny) == (0, 0, 0)
    gap = root / "gap.mid"
    midi(gap, bytes.fromhex("00 90 3c 64 60 80 3c 00 96 40 90 3c 64 60 80 3c 00 00 ff 2f 00"))
    silent = render("gap", [("-", "soft", "gap", "out")], "--quality-gate", "custom",
                    "--qa-max-silence-seconds", "1", expected=2)
    assert "excessive_audio_silence" in silent.stderr and transfer(silent) == (0, 0, 0)

    stereo = render("stereo", rows, "--tail-policy", "orig", "--group-volume", "mean",
                    "--quality-gate", "clayengine", expected=2)
    assert "quiet_performance" in stereo.stderr and read_wav(root / "stereo/soft.wav")[0] == 2
    (root / "hot.sfz").write_text((root / "loud.sfz").read_text().strip() + " volume=30\n", encoding="ascii")
    clipped = render("clipped", [("-", "hot", "melody", "out")], "--tail-policy", "orig",
                     "--quality-gate", "clayengine", expected=2)
    assert "excessive_audio_clipping" in clipped.stderr and transfer(clipped) == (0, 0, 0)

    with wave.open(str(root / "spiky.wav"), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(struct.pack("<h", 12000) * (RATE // 1000) + struct.pack("<h", 100) * RATE)
    (root / "spiky.sfz").write_text((root / "soft.sfz").read_text().strip().replace("soft.wav", "spiky.wav")
                                    + " volume=10\n", encoding="ascii")
    limited = render("peak-limit", [("pair", "loud", "melody", "loud"), ("pair", "spiky", "melody", "spiky")],
                     "--group-volume", "mean")
    levels = [list(map(float, line.split("\t")[2:])) for line in limited.stdout.splitlines()
              if line.startswith("VOLUME\t")]
    assert levels[0][1] == levels[1][1] < (levels[0][0] + levels[1][0]) / 2
    assert max(abs(x) for x in pcm(root / "peak-limit/spiky.wav")) == round(32767 * 10 ** (-1 / 20))

    # Oversized group must survive the 64-job CUDA boundary. The unrelated tracks
    # complete independently and neither rejected output enters the waiting area.
    large = [("-", "soft", "melody", "solo")]
    large += [("large", "loud" if i % 2 else "soft", "melody", f"out-{i}") for i in range(65)]
    large += [("large", "quiet", "melody", "quiet"), ("small", "soft", "melody", "small-1"),
              ("small", "loud", "melody", "small-2")]
    flags = ("--group-volume", "mean", "--quality-gate", "clayengine")
    device = render("device", large, *flags, expected=2)
    host = render("host", large, *flags, "--group-wait-mib", "0", expected=2)
    device_bytes, device_spill, device_wait = transfer(device)
    host_bytes, host_spill, host_wait = transfer(host)
    assert device_bytes == host_bytes > device_wait > 0
    assert (device_spill, host_wait, host_spill) == (0, 0, device_wait)
    assert device.stdout.index("solo.wav") < device.stdout.index("out-0.wav")
    for _, _, _, output in large:
        if output != "quiet":
            assert (root / f"device/{output}.wav").read_bytes() == (root / f"host/{output}.wav").read_bytes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clay", type=Path, required=True)
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--sndfile", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="claysampler-group-") as temp:
        check(args, Path(temp))
    print("Grouped audio checks passed")
