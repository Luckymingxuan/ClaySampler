"""Exercise the three tail policies through the CUDA CLI with generated inputs."""

import argparse
import struct
import subprocess
import tempfile
import wave
from pathlib import Path


RATE = 44100


def midi(path, data):
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
                     + b"MTrk" + struct.pack(">I", len(data)) + data)


def read_wav(path):
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        frames = stream.getnframes()
        return channels, frames, stream.readframes(frames)


def check(options, root):
    sample = root / "sample.wav"
    with wave.open(str(sample), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(struct.pack("<h", 1600) * (RATE * 2 // 5)
                           + struct.pack("<h", 12) * (RATE * 3))
    sfz = root / "sample.sfz"
    sfz.write_text("<region> sample=sample.wav key=60 loop_mode=one_shot amp_veltrack=0\n", encoding="ascii")
    source = root / "source.mid"
    # Note-off at 0.1 s, followed by an EOT gap of 5 s.
    midi(source, bytes.fromhex("00 90 3c 64 60 80 3c 00 a5 40 ff 2f 00"))

    def render(name, *flags, inputs=(source,), preset=sfz, expected=0):
        output = root / name
        command = [options.clay.resolve(), "--sfz", preset, "--output", output, *flags]
        for key in ("nvrtc", "sndfile"):
            value = getattr(options, key)
            if value is not None:
                command += [f"--{key}", value.resolve()]
        result = subprocess.run(command + list(inputs), capture_output=True, text=True, check=False)
        if result.returncode != expected:
            raise AssertionError(result.stdout + result.stderr)
        return output, result

    natural, natural_log = render("default")
    explicit, _ = render("natural", "--tail-policy", "natural")
    natural_wave = natural / "source.wav"
    assert natural_wave.read_bytes() == (explicit / "source.wav").read_bytes()
    channels, frames, samples = read_wav(natural_wave)
    assert (channels, frames) == (1, round(.65 * RATE))
    assert struct.unpack_from("<h", samples, (frames - 5000) * 2)[0] > 3
    assert samples[-2:] == b"\0\0"
    peaks = [abs(value) for (value,) in struct.iter_unpack("<h", samples)]
    audible = [i for i, peak in enumerate(peaks) if peak > 3]
    qc = next(line.split("\t")[2:] for line in natural_log.stdout.splitlines() if line.startswith("OK\t"))
    assert list(map(int, qc)) == [frames, max(peaks), audible[0], audible[-1]]

    orig, _ = render("orig", "--tail-policy", "orig")
    assert read_wav(orig / "source.wav")[:2] == (2, round(5.1 * RATE))

    custom, _ = render("custom", "--tail-policy", "custom", "--tail-max-seconds", "1", "--tail-fade-ms", "100")
    dry, _ = render("custom-dry", "--tail-policy", "custom", "--tail-max-seconds", "1", "--tail-fade-ms", "0")
    channels, frames, samples = read_wav(custom / "source.wav")
    _, _, dry_samples = read_wav(dry / "source.wav")
    assert (channels, frames) == (2, round(1.1 * RATE) + 1)
    assert samples[:-4410 * 4] == dry_samples[:-4410 * 4]
    assert samples[-4:] == b"\0\0\0\0" and dry_samples[-4:] != samples[-4:]

    uncut, _ = render("custom-uncut", "--tail-policy", "custom", "--tail-max-seconds", "4")
    channels, frames, samples = read_wav(uncut / "source.wav")
    assert (channels, frames) == (2, round(3.4 * RATE))
    assert samples == read_wav(orig / "source.wav")[2][:frames * 4]

    limited, result = render("limited", "--max-audio-seconds", "0.5", expected=2)
    assert "tail or audio duration limit" in result.stderr
    assert not (limited / "source.wav").exists()

    loop = root / "loop.sfz"
    loop.write_text("<region> sample=sample.wav key=60 loop_mode=loop_continuous loop_start=0 "
                    "loop_end=800 ampeg_release=0.1 amp_veltrack=0\n", encoding="ascii")
    regular, held, later, sostenuto = (root / f"{name}.mid" for name in ("regular", "held", "later", "sostenuto"))
    midi(regular, bytes.fromhex("00 90 3c 64 60 80 3c 00 00 ff 2f 00"))
    midi(held, bytes.fromhex("00 b0 40 7f 00 90 3c 64 60 80 3c 00 00 b0 78 00 a5 40 ff 2f 00"))
    midi(later, bytes.fromhex("00 b0 40 7f 00 90 3c 64 60 80 3c 00 82 68 b0 40 00 00 ff 2f 00"))
    midi(sostenuto, bytes.fromhex("00 90 3c 64 30 b0 42 7f 30 80 3c 00 82 68 b0 42 00 00 ff 2f 00"))
    pedals, _ = render("pedals", inputs=(regular, held, later, sostenuto), preset=loop)
    assert (pedals / "regular.wav").read_bytes() == (pedals / "held.wav").read_bytes()
    assert (pedals / "later.wav").read_bytes() == (pedals / "sostenuto.wav").read_bytes()
    assert read_wav(pedals / "later.wav")[1] > read_wav(pedals / "held.wav")[1] + RATE // 4
    assert read_wav(pedals / "held.wav")[1] % 1024 == 0

    processed = root / "processed.sfz"
    processed.write_text(loop.read_text(encoding="ascii").strip()
                         + " fil_type=lpf_2p cutoff=1200 lfo1_freq=3 lfo1_pitch=120\n", encoding="ascii")
    filtered, _ = render("filtered", inputs=(regular, later), preset=processed)
    assert (filtered / "regular.wav").read_bytes() != (pedals / "regular.wav").read_bytes()

    hot = root / "hot.sfz"
    centered = sfz.read_text(encoding="ascii").strip() + " pan_oncc10=0"
    hot.write_text(centered + " volume=33\n", encoding="ascii")
    repaired, _ = render("repaired", preset=hot)
    _, _, repaired_samples = read_wav(repaired / "source.wav")
    repaired_peak = max(abs(value) for (value,) in struct.iter_unpack("<h", repaired_samples))
    assert abs(repaired_peak - round(32768 * 10 ** (-1 / 20))) <= 1, repaired_peak
    hot.write_text(centered + " volume=42\n", encoding="ascii")
    rejected, result = render("too-hot", preset=hot, expected=2)
    assert "excessive_peak" in result.stderr and not (rejected / "source.wav").exists()

    short_sample = root / "short.wav"
    with wave.open(str(short_sample), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(struct.pack("<h", 1600) * (RATE * 2 // 5))
    short_preset = root / "short.sfz"
    short_preset.write_text(sfz.read_text(encoding="ascii").replace("sample.wav", "short.wav"), encoding="ascii")
    gap = root / "gap.mid"
    midi(gap, bytes.fromhex("00 90 3c 64 8b 20 80 3c 00 00 ff 2f 00"))
    padded, _ = render("padded", inputs=(gap,), preset=short_preset)
    channels, frames, pcm = read_wav(padded / "gap.wav")
    assert (channels, frames) == (1, 65 * 1024)
    assert not any(pcm[(RATE * 2 // 5) * 2:])

    many = [root / f"batch-{i}.mid" for i in range(66)]
    for path in many:
        path.write_bytes(regular.read_bytes())
    many[64].write_bytes(b"invalid MIDI")
    output, result = render("pipeline", "--passes", "2", inputs=many, preset=short_preset, expected=2)
    assert len(list(output.glob("*.wav"))) == 65
    assert sum(line.startswith("OK\t") for line in result.stdout.splitlines()) == 130
    assert (output / "batch-0.wav").read_bytes() == (output / "batch-65.wav").read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clay", type=Path, required=True)
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--sndfile", type=Path)
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tail-", dir=Path.cwd()) as directory:
        check(options, Path(directory))
    print("Tail policy checks passed")


if __name__ == "__main__":
    main()
