# Validation

The comparison harness renders the same inputs through ClaySampler and a pinned sfizz 1.2.3 executable. It writes the original WAVs, MIDI inputs, an offline A/B page, and a JSON report. The report includes hashes of both executables. Use a new output directory for each run.

The harness needs Python with NumPy. Specify existing SFZ paths. SFZ presets and sample libraries are read in place. All rendered files and synthetic MIDI cases are written to the requested output directory.

```sh
python tests/compare.py --clay build/claysampler --sfizz path/to/sfizz_render \
  --instrument clarinet=path/to/clarinetsus.sfz \
  --instrument piano=path/to/grand_piano_steinway_b.sfz \
  --midi piano=path/to/held_out.mid --output build/ab
```

Use `--nvrtc`, `--sndfile`, and `--device` to pass runtime options to ClaySampler. On Windows use the actual executable paths, which may be in a `Release` subdirectory.

## Measurements

- Both sides output 44.1 kHz stereo PCM16. The harness explicitly selects `--tail-policy orig`. The sfizz CLI runs with quality 8 (48-tap sinc), a 1024-frame block, and a 256-voice limit. Sixteen workers each launch sfizz separately for their MIDI file.
- Cold CUDA includes preset metadata, CUDA context, NVRTC compilation, sample decode and upload, render, download, WAV write and quality check. Warm CUDA reuses the sample bank and module. The sfizz repeat timing still starts a fresh process per MIDI. The OS file cache is left in place for both renderers.
- A pair passes the automatic comparison when onset and last audible frame are within 12 ms and 80 ms, integrated energy is within 0.5 dB, relative waveform error is at most 18% after one sub-ms global alignment, and clipping does not materially increase. The tail check uses a threshold of 16 PCM16 units (about -66 dBFS). The report also records the end delta at three units.
- Synthetic cases cover velocity layers, repeated notes, pedal, bend and CC automation. The harness accepts held-out MIDI via `--midi`. Review the original WAVs in the A/B page to judge audible changes that these checks do not capture.
- Peak CUDA allocations cover sample storage, the sinc table, and explicitly allocated device buffers. CUDA event timing covers synthesis and GPU finishing. Planning and upload measurements can overlap initialization, rendering, and WAV writing. Use pass wall time for throughput comparisons.

## First recorded run (`aedaa4d`)

Windows x86-64, RTX 4080 Laptop GPU, NVRTC 12.4. Each preset used 16 generated cases and two held-out GigaMIDI clips. All 36 pairs passed the automatic checks. The largest integrated-energy difference was under 0.004 dB and the largest tail difference at the 16-unit threshold was 11.4 ms.

| Preset | sfizz first batch | CUDA cold | sfizz repeat | CUDA warm | CUDA warm kernel | Peak CUDA allocations |
|---|---:|---:|---:|---:|---:|---:|
| VSCO2 Clarinet Sustain | 1.487 s | 0.448 s | 1.487 s | 0.112 s | 0.010 s | 100 MiB |
| VCSL Grand Piano Steinway B | 5.661 s | 1.350 s | 5.413 s | 0.130 s | 0.034 s | 915 MiB |

These are whole-batch timings for the stated workloads. The same kernel also compiled for `sm_75`, `sm_86`, `sm_89` and `compute_90` with NVRTC 12.4. Audio and performance were measured on the RTX 4080 Laptop GPU.

The subsequent compatibility work reran the 36 audio comparisons. See [compatibility](compatibility.md#additional-audio-checks) for those results and the additional preset checks.

## Tail policies

`natural` follows the ClayEngine `sfizz_native_mono_v7` policy. It starts from the last complete MIDI note, includes explicit pedal releases that extend the performance, ignores trailing End-of-Track gaps and terminal panic commands, and releases CC64, CC66 and CC69 at the prepared end. The prepared performance uses the ClayEngine controller defaults and a 0 dB synth master setting.

The first 1024-frame stereo float block at or after the final event with mean-square power at most `1e-12` sets the native output endpoint. The search is bounded by 120 seconds after that event and an optional `--max-audio-seconds` limit. The renderer repairs a clipping float mix to −1 dBFS with at most 5 dB of attenuation, then rounds each channel to PCM16 and averages the two with ties-to-even rounding.

The final mono pass uses 50 ms windows. Its RMS threshold is 30 dB below the strongest window, clamped to −78 through −60 dBFS, with a peak guard at four times that threshold. At least two seconds of weak tail are required to crop. It keeps 250 ms after the last active window and fades the final 100 ms with a half cosine. The policy runs on ClaySampler's own synthesis output, so durations and samples can differ from sfizz for individual presets.

Run the generated CLI regression cases with:

```sh
python tests/check_tail.py --clay build/claysampler
```

Pass `--nvrtc` and `--sndfile` for explicit runtime library paths.

On the RTX 4080 Laptop GPU, `orig` reproduced all 36 clarinet and piano WAVs byte for byte against the executable from `cba2724`. Against ClayEngine's sfizz 1.2.3 path, `natural` matched the frame count for all 36 clips. The final frame above 16 PCM16 units differed by at most two samples. The separate long weak-tail fixture reached the same 1.050 s output duration on both paths, from a 4.040 s reference native render.

At `7a4598e`, for batches of 18 clips, median warm times over three passes were 0.181 s for clarinet and 0.222 s for piano in `natural`, compared with 0.113 s and 0.150 s in `orig`. These timings include WAV writeout. ClayEngine's 16-worker reference took 1.80 s and 6.18 s respectively, including its per-clip MIDI preparation and sfizz setup.

## CUDA finishing pipeline

The comparison uses real MIDI from the ClayEngine VCSL v3 manifests, the same preset and `natural` arguments on both binaries, and the RTX 4080 Laptop GPU with NVRTC 12.4. The baseline is `7a4598e`. Warm values are medians of ten passes across two invocations per binary, after the first pass in each invocation. Cold values are medians of three fresh processes and include startup and WAV writing. The operating system's file cache was kept between runs.

| Workload | Baseline warm | Pipeline warm | Baseline cold | Pipeline cold |
|---|---:|---:|---:|---:|
| Steinway B, 1 clip | 0.022 s | 0.007 s | 0.986 s | 0.858 s |
| Steinway B, 16 clips | 0.420 s | 0.116 s | 1.768 s | 1.081 s |
| Steinway B, 63 clips | 1.669 s | 0.308 s | 3.494 s | 1.649 s |
| Concert Harp, 16 clips | 1.582 s | 1.194 s | | |
| Marimba, 16 clips | 1.236 s | 0.900 s | | |

The performance workloads produced byte-identical WAVs against the baseline. The 36 clarinet and piano regression clips matched byte for byte in both `natural` and `orig`. The generated CLI checks cover custom fade, peak repair, weak-tail trim, silent padding, filtered pitch modulation, and a rejected MIDI across repeated batches. NVRTC 12.4 compiled both kernel variants for `sm_75`, `sm_86`, `sm_89`, and `compute_90`.

The 63-clip piano warm CUDA interval grew from 0.179 s to 0.199 s, including GPU tail finishing. The total pass fell from 1.669 s to 0.308 s. Peak explicit device allocations rose from 1196 MiB to 1471 MiB. The host download now carries mono PCM16 at two bytes per copied frame in `natural`, compared with eight bytes for the prior stereo float path.

## Preset audit

The audit uses the host-only ClaySampler loader and the sfizz C API on the same paths. It records accepted region counts, ClaySampler rejection reasons and unknown opcodes reported by sfizz. The result measures loading and sample resolution. Use the audio comparison harness for MIDI-specific behavior and sound.

```sh
python tests/audit_presets.py --clay build/claysampler --sfizz-library path/to/sfizz.dll \
  --root path/to/instruments --pattern '*/*/presets/*.sfz' --output build/sfz-audit.json
```
