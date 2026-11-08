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

- Both sides output 44.1 kHz stereo PCM16. The sfizz CLI runs with quality 8 (48-tap sinc), a 1024-frame block, and a 256-voice limit. Sixteen workers each launch sfizz separately for their MIDI file.
- Cold CUDA includes preset metadata, CUDA context, NVRTC compilation, sample decode and upload, render, download, WAV write and quality check. Warm CUDA reuses the sample bank and module. The sfizz repeat timing still starts a fresh process per MIDI. The OS file cache is left in place for both renderers.
- A pair passes the automatic comparison when onset and last audible frame are within 12 ms and 80 ms, integrated energy is within 0.5 dB, relative waveform error is at most 18% after one sub-ms global alignment, and clipping does not materially increase. The tail check uses a threshold of 16 PCM16 units (about -66 dBFS). The report also records the end delta at three units.
- Synthetic cases cover velocity layers, repeated notes, pedal, bend and CC automation. The harness accepts held-out MIDI via `--midi`. Review the original WAVs in the A/B page to judge audible changes that these checks do not capture.
- Peak CUDA allocations refer to sample storage, the sinc table, and explicitly allocated render buffers. They exclude the driver and context. CUDA kernel time is recorded separately.

## First recorded run (`aedaa4d`)

Windows x86-64, RTX 4080 Laptop GPU, NVRTC 12.4. Each preset used 16 generated cases and two held-out GigaMIDI clips. All 36 pairs passed the automatic checks. The largest integrated-energy difference was under 0.004 dB and the largest tail difference at the 16-unit threshold was 11.4 ms.

| Preset | sfizz first batch | CUDA cold | sfizz repeat | CUDA warm | CUDA warm kernel | Peak CUDA allocations |
|---|---:|---:|---:|---:|---:|---:|
| VSCO2 Clarinet Sustain | 1.487 s | 0.448 s | 1.487 s | 0.112 s | 0.010 s | 100 MiB |
| VCSL Grand Piano Steinway B | 5.661 s | 1.350 s | 5.413 s | 0.130 s | 0.034 s | 915 MiB |

These are whole-batch timings for the stated workloads. The same kernel also compiled for `sm_75`, `sm_86`, `sm_89` and `compute_90` with NVRTC 12.4. Audio and performance were measured on the RTX 4080 Laptop GPU.

The subsequent compatibility work reran the 36 audio comparisons. See [compatibility](compatibility.md#additional-audio-checks) for those results and the additional preset checks.
