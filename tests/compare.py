"""Render fixed MIDI cases with ClaySampler and sfizz, then write a local A/B report."""

import argparse
import concurrent.futures
import hashlib
import html
import json
import math
import random
import shutil
import statistics
import struct
import subprocess
import time
import wave
from pathlib import Path
from urllib.parse import quote

import numpy as np


RATE = 44100
PPQ = 480
SUSTAIN = 64


def variable(value):
    data = [value & 127]
    while value > 127:
        value >>= 7
        data.append((value & 127) | 128)
    return bytes(reversed(data))


def fixture(path, case, seed, piano):
    rng = random.Random(seed)
    events = [(0, bytes.fromhex("ff510307a120"))]
    measure = 4 * PPQ
    keys = (48, 52, 55, 60, 64, 67, 72, 76) if piano else (58, 60, 62, 65, 67, 69, 72, 74)
    for bar in range(6 + seed % 3):
        start = 120 + bar * measure
        if case == "pedal":
            events.extend([(start - 20, bytes((0xB0, SUSTAIN, 127))),
                           (start + measure - 95, bytes((0xB0, SUSTAIN, 0)))])
        for beat in range(4):
            offset = start + beat * PPQ
            key = rng.choice(keys)
            if case == "repeat":
                key = keys[3 + bar % 2]
            notes = [key]
            if piano and case in {"pedal", "layers"}:
                notes.extend([key - 12, key + 7])
            velocity = (48, 76, 105, 119)[beat]
            duration = 135 if case == "repeat" else 390
            for note in notes:
                events.extend([(offset, bytes((0x90, note, velocity))),
                               (offset + duration, bytes((0x80, note, 0)))])
            if case == "bend":
                events.extend([(offset + 120, bytes((0xE0, 0, 80 + (beat % 2) * 8))),
                               (offset + 300, bytes((0xE0, 0, 64)))])
            if case == "controllers":
                events.extend([(offset + 110, bytes((0xB0, 7, 72 + beat * 10))),
                               (offset + 190, bytes((0xB0, 10, 26 + beat * 24))),
                               (offset + 270, bytes((0xB0, 11, 92 + beat * 9)))])
    end = max(tick for tick, _ in events) + 60
    events.append((end, bytes.fromhex("ff2f00")))
    track = bytearray()
    previous = 0
    for tick, event in sorted(events, key=lambda event: event[0]):
        track.extend(variable(tick - previous))
        track.extend(event)
        previous = tick
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, PPQ) +
                     b"MTrk" + struct.pack(">I", len(track)) + track)


def pcm(path):
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (2, 2, RATE):
            raise ValueError(f"unexpected WAV format in {path}")
        frames = stream.getnframes()
        samples = np.frombuffer(stream.readframes(frames), dtype="<i2").reshape(-1, 2)
        if len(samples) != frames or not len(samples):
            raise ValueError(f"empty or truncated WAV in {path}")
        return samples


def qc(samples):
    peaks = np.max(np.abs(samples.astype(np.int32)), axis=1)
    indices = np.flatnonzero(peaks > 3)
    if not len(indices):
        raise ValueError("silent WAV")
    peak = int(peaks.max())
    return {
        "first": int(indices[0]),
        "last": int(indices[-1]),
        "last_above_16": int(np.flatnonzero(peaks > 16)[-1]) if peak > 16 else int(indices[-1]),
        "peak": peak,
        "clip_fraction": float(np.count_nonzero(peaks >= 32767) / len(peaks)),
    }


def sfizz_job(binary, sfz, midi, output):
    wav = output / midi.with_suffix(".wav").name
    result = subprocess.run(
        [binary, "--sfz", sfz, "--midi", midi, "--wav", wav, "--samplerate", str(RATE),
         "--quality", "8", "--blocksize", "1024", "--polyphony", "256"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    samples = pcm(wav)
    if not (np.any(samples > 3) or np.any(samples < -3)):
        raise ValueError("silent WAV")


def baseline(binary, sfz, midis, output, passes, workers):
    output.mkdir()
    times = []
    failures = {}
    for _ in range(passes):
        start = time.perf_counter()
        failures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(sfizz_job, binary, sfz, midi, output): midi for midi in midis}
            for task in concurrent.futures.as_completed(futures):
                try:
                    task.result()
                except (ValueError, OSError, RuntimeError) as error:
                    failures[futures[task].stem] = str(error)
        times.append(time.perf_counter() - start)
    return times, failures


def cuda(options, sfz, midis, output):
    command = [options.clay, "--sfz", sfz, "--output", output, "--samplerate", str(RATE),
               "--passes", str(options.passes)]
    for name in ("nvrtc", "sndfile", "device"):
        value = getattr(options, name)
        if value is not None:
            command.extend([f"--{name}", str(value)])
    start = time.perf_counter()
    result = subprocess.run(command + midis, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", check=False)
    total = time.perf_counter() - start
    initialization, device, passes, rejected = 0, "", [], {}
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "DEVICE":
            device = parts[1]
        elif parts[0] == "INIT":
            initialization = float(parts[1])
        elif parts[0] == "PASS":
            passes.append({"seconds": float(parts[2]), "accepted": int(parts[3]),
                           "allocated_bytes": int(parts[4]), "prepare_seconds": float(parts[5]),
                           "kernel_seconds": float(parts[6])})
    if len(passes) != options.passes or result.returncode not in {0, 2}:
        raise RuntimeError(result.stdout + result.stderr)
    for line in result.stderr.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[0] == "REJECT":
            rejected[Path(parts[1]).stem] = parts[2]
    return device, initialization, total, passes, rejected


def compare(cpu, gpu):
    a, b = pcm(cpu), pcm(gpu)
    frames_a, frames_b = len(a), len(b)
    qa, qb = qc(a), qc(b)
    power_a = float(np.sum(np.square(a, dtype=np.float64)))
    power_b = float(np.sum(np.square(b, dtype=np.float64)))
    onset = (qb["first"] - qa["first"]) * 1000 / RATE
    tail = (qb["last_above_16"] - qa["last_above_16"]) * 1000 / RATE
    level = 10 * math.log10(power_b / power_a)
    # Allow a single sub-ms scheduling offset. Individual note drift remains in the residual.
    first = max(qa["first"], qb["first"]) + 40
    last = min(first + RATE, len(a) - 40, len(b) - 40)
    if last - first > 256:
        x = a[first:last:7].astype(np.float64)
        shift = min(range(-32, 33), key=lambda s: np.sum(
            np.square(x - b[first + s:last + s:7])))
    else:
        shift = 0
    a = a[max(0, -shift):]
    b = b[max(0, shift):]
    overlap = min(len(a), len(b))
    error = np.sum(np.square(a[:overlap].astype(np.float64) - b[:overlap]))
    for rest in (a[overlap:], b[overlap:]):
        error += np.sum(np.square(rest, dtype=np.float64))
    error_ratio = float(math.sqrt(error / power_a))
    return {
        "cpu_frames": frames_a,
        "gpu_frames": frames_b,
        "onset_delta_ms": onset,
        "last_audible_delta_ms": tail,
        "last_above_3_delta_ms": (qb["last"] - qa["last"]) * 1000 / RATE,
        "energy_delta_db": level,
        "alignment_frames": shift,
        "relative_wave_error": error_ratio,
        "cpu_clip_fraction": qa["clip_fraction"],
        "gpu_clip_fraction": qb["clip_fraction"],
        "passed": (abs(onset) <= 12 and abs(tail) <= 80 and abs(level) <= .5 and
                   error_ratio <= .18 and qb["clip_fraction"] <= qa["clip_fraction"] + .001),
    }


def report(root, options, groups):
    esc = html.escape
    rows, cases = [], []
    for group in groups:
        label = esc(group["name"])
        passed = group["matched"]
        attempted = group["attempted"]
        cpu = statistics.median(group["sfizz_seconds"][1:] or group["sfizz_seconds"])
        gpu = statistics.median(x["seconds"] for x in group["cuda_passes"][1:] or group["cuda_passes"])
        cpu_accepted = attempted - len(group["sfizz_rejected"])
        rows.append(
            f"<tr><td>{label}<td>{passed}/{attempted}<td>{group['sfizz_seconds'][0]:.3f} s"
            f"<td>{group['cuda_init_seconds'] + group['cuda_passes'][0]['seconds']:.3f} s"
            f"<td>{cpu:.3f} s<td>{gpu:.3f} s<td>{cpu_accepted / cpu:.2f}"
            f"<td>{passed / gpu:.2f}<td>{max(x['allocated_bytes'] for x in group['cuda_passes']) / 2**20:.1f} MiB"
        )
        for item in group["cases"]:
            if "metrics" not in item:
                cases.append(f"<tr><td>{label}<td>{esc(item['name'])}<td colspan=5>{esc(item['error'])}")
                continue
            m = item["metrics"]
            sources = [quote(item[key]) for key in ("sfizz", "cuda")]
            cases.append(
                f"<tr class=pair data-first='{sources[0]}' data-second='{sources[1]}'>"
                f"<td>{label}<td>{esc(item['name'])}<td>{'PASS' if m['passed'] else 'REVIEW'}"
                f"<td>{m['energy_delta_db']:+.2f} dB<td>{m['relative_wave_error']:.1%}"
                f"<td><label><span></span><br><audio controls preload=none></audio></label>"
                f"<td><label><span></span><br><audio controls preload=none></audio></label>"
            )
    page = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ClaySampler A/B</title>
<style>
body {{ font: 15px system-ui, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; margin-bottom: 22px; }}
th, td {{ border: 1px solid #aaa; padding: 6px; text-align: left; }}
audio {{ width: min(215px, 30vw); height: 35px; }}
</style>
<h1>ClaySampler A/B</h1>
<p>{esc(groups[0]['device'])} · sfizz 1.2.3 · {options.workers} CPU workers · 48-tap sinc · {RATE} Hz stereo PCM16</p>
<table><tr><th>Instrument<th>Pairs passing checks<th>sfizz first batch<th>CUDA cold
<th>sfizz repeat<th>CUDA warm<th>sfizz valid WAV/s<th>CUDA matched WAV/s<th>CUDA peak allocations
{''.join(rows)}</table>
<p>Times include WAV writing and per-render checks. CUDA cold includes initialization.
CUDA warm reuses samples in VRAM. sfizz starts a fresh process per MIDI on each pass.
Pair checks cover onset, audible end, energy, clipping, and relative waveform error. See <a href="results.json">results.json</a>.</p>
<p><label><input id="blind" type="checkbox" checked> Hide renderer names</label>
· The A/B order is randomized. Uncheck to reveal the renderer without changing the order.</p>
<table><tr><th>Instrument<th>Case<th>Pair check<th>Energy Δ<th>Waveform error<th>Player 1<th>Player 2
{''.join(cases)}</table>
<script>
const blind = document.querySelector("#blind");
const labels = [];
for (const row of document.querySelectorAll(".pair")) {{
  const tracks = [[row.dataset.first, "sfizz"], [row.dataset.second, "ClaySampler"]];
  if (Math.random() < 0.5) tracks.reverse();
  row.querySelectorAll("audio").forEach((audio, i) => {{
    audio.src = tracks[i][0];
    labels.push([audio.parentElement.querySelector("span"), "AB"[i], tracks[i][1]]);
  }});
}}
function showLabels() {{
  for (const [label, letter, renderer] of labels) {{
    label.textContent = blind.checked ? letter : letter + " · " + renderer;
  }}
}}
blind.addEventListener("change", showLabels);
showLabels();
</script>
</html>
"""
    (root / "index.html").write_text(page, encoding="utf-8")


def group(options, name, sfz):
    root = options.output / name
    input_dir = root / "midi"
    input_dir.mkdir(parents=True)
    midis = []
    for index in range(options.cases):
        case = ("layers", "repeat", "pedal", "bend", "controllers")[index % 5]
        midi = input_dir / f"{case}-{index + 1:02}.mid"
        fixture(midi, case, index, "piano" in name.lower())
        midis.append(midi)
    for label, source in options.midi:
        if label == name:
            destination = input_dir / f"real-{source.stem}.mid"
            if destination.exists():
                raise ValueError(f"duplicate input {destination}")
            shutil.copyfile(source, destination)
            midis.append(destination)
    cpu_dir, gpu_dir = root / "sfizz", root / "cuda"
    cpu_times, cpu_rejected = baseline(options.sfizz, sfz, midis, cpu_dir, options.passes, options.workers)
    device, init, wall, gpu_passes, gpu_rejected = cuda(options, sfz, midis, gpu_dir)
    pairs = []
    for midi in midis:
        wav = midi.with_suffix(".wav").name
        a, b = cpu_dir / wav, gpu_dir / wav
        item = {"name": midi.stem, "midi": midi.relative_to(options.output).as_posix()}
        try:
            if midi.stem in cpu_rejected:
                raise ValueError(cpu_rejected[midi.stem])
            if midi.stem in gpu_rejected:
                raise ValueError(gpu_rejected[midi.stem])
            item["metrics"] = compare(a, b)
            item["sfizz"] = a.relative_to(options.output).as_posix()
            item["cuda"] = b.relative_to(options.output).as_posix()
        except (OSError, ValueError, wave.Error) as error:
            item["error"] = str(error)
        pairs.append(item)
    return {
        "name": name, "sfz": str(sfz), "device": device, "attempted": len(midis),
        "matched": sum(p.get("metrics", {}).get("passed", False) for p in pairs),
        "sfizz_seconds": cpu_times, "sfizz_rejected": cpu_rejected,
        "cuda_init_seconds": init, "cuda_wall_seconds": wall,
        "cuda_passes": gpu_passes, "cuda_diagnostics": gpu_rejected, "cases": pairs,
    }


def named_path(value):
    name, sep, path = value.partition("=")
    if not sep or not name or not all(c.isascii() and (c.isalnum() or c in "-_") for c in name):
        raise argparse.ArgumentTypeError("expected label=path with a simple ASCII label")
    return name, Path(path).resolve(strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clay", type=Path, required=True)
    parser.add_argument("--sfizz", type=Path, required=True)
    parser.add_argument("--instrument", type=named_path, action="append", required=True)
    parser.add_argument("--midi", type=named_path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--sndfile", type=Path)
    parser.add_argument("--device", type=int)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--passes", type=int, default=3)
    options = parser.parse_args()
    if min(options.workers, options.passes, options.cases) < 1:
        parser.error("workers, passes, and cases must be positive")
    labels = {name for name, _ in options.instrument}
    if len(labels) != len(options.instrument) or any(name not in labels for name, _ in options.midi):
        parser.error("instrument labels must be unique and MIDI labels must refer to an instrument")
    options.clay = options.clay.resolve(strict=True)
    options.sfizz = options.sfizz.resolve(strict=True)
    options.output.mkdir(parents=True)
    groups = []
    for name, sfz in options.instrument:
        results = group(options, name, sfz)
        groups.append(results)
        print(name, results["matched"], "/", results["attempted"], flush=True)
    metadata = {
        "workers": options.workers, "rate": RATE, "sfizz_resampling_quality": 8,
        "sfizz_sha256": hashlib.sha256(options.sfizz.read_bytes()).hexdigest(),
        "clay_sha256": hashlib.sha256(options.clay.read_bytes()).hexdigest(),
        "groups": groups,
    }
    (options.output / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report(options.output, options, groups)


if __name__ == "__main__":
    main()
