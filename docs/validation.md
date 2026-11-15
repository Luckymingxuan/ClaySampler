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

## GPU quality and group volume

```sh
python tests/check_group_audio.py --clay build/claysampler
```

The generated cases cover opt-in quality, custom thresholds, quiet and clipped rejection, duration and body silence, peak-limited gain, stereo output, an incomplete group and a group larger than one CUDA batch. The oversized group produces identical WAVs with device retention and with forced host staging. A fully rejected batch reports zero downloaded PCM. NVRTC 12.4 compiled both output modes with the two features enabled for `sm_75`, `sm_86`, `sm_89` and `compute_90`.

On the RTX 4080 Laptop GPU, 16 real VCSL melodies were each rendered with Steinway B, Concert Harp and Marimba. The CPU reference ran ClayEngine's quality check and group RMS rules on ClaySampler's unadjusted WAVs. It agreed with the GPU's three rejections and all 45 accepted PCM streams matched byte for byte after gain. A separate 16-file comparison of the default single-preset mode with `8d437c2` also matched byte for byte.

The median warm pass was about 0.91 s with both options off and 0.93 s with both on, including WAV writes. The separate CPU check and gain computation on existing WAVs took about 1.52 s, excluding writes of the adjusted WAVs. This compares the post-processing paths on the same synthesis output. The GPU path downloaded 136.2 MiB of final PCM per pass, used no host staging, and peaked at about 506 MiB of explicit device allocations.

### ClayEngine batch comparison

A separate run submitted the same 48 jobs to ClayEngine's `render_planned` using sfizz 1.2.3 with 16 CPU workers, and to ClaySampler with both GPU options enabled. Both paths rendered at 44.1 kHz, finished tails, ran the audio gate, matched group volume and wrote WAVs. The inputs were fixed in advance, so the timing excludes candidate selection and replacement. The CPU timing starts with ClayEngine already loaded. Its benchmark wrapper bypasses only the suppression of sfizz's freewheeling log line.

| Path | Batch time | Speedup over ClayEngine |
| --- | ---: | ---: |
| ClayEngine, 16 CPU workers, one run | 32.75 s | 1× |
| ClaySampler, median of two fresh processes including startup | 2.02 s | 16.2× |
| ClaySampler, median of three passes reusing the process and sample cache | 0.94 s | 34.9× |

Both renderers rejected the same three jobs and wrote the same 45 job outputs. The batch comparison checked acceptance and output counts. The byte-for-byte comparison above used the same synthesis output to validate post-processing.

## Dynamic CUDA batches and kernel cache

On an RTX 4080 Laptop GPU, ClaySampler v0.1.0 and the updated renderer each processed the same 768 GigaMIDI and VSCO2 jobs in twelve fresh 64-job invocations per round. The order was baseline, updated with an empty kernel cache, updated with a populated cache, then baseline again. Each round produced 652 byte-identical accepted WAVs and identical rejection reasons.

| Metric | v0.1.0 | Updated |
| --- | ---: | ---: |
| Mean round wall time | 17.06 s | 10.25 s |
| Mean GPU utilization | 32.9% | 46.8% |
| GPU utilization samples at or below 20% | 54.8% | 20.0% |

The wall times include process startup, rendering, quality checks and WAV writes. They exclude ClayEngine's upstream selection and replacement work. Accepted WAV throughput increased by 66.4%, and wall time fell by 39.9%. Device-wide utilization was sampled with `nvidia-smi` at 200 ms intervals. CUDA event time was about 2.94–2.98 seconds per round across both versions. The main measured gain was shorter initialization, whose per-batch median fell from about 0.87 seconds to 0.16–0.18 seconds.

A separate 24-job synthetic mix of small and medium renders fell from 0.96 to 0.66 seconds per pass at the median over five passes, with byte-identical WAVs. A three-job case with one oversized render failed at the v0.1.0 scratch budget check and produced all three WAVs with dynamic batching. A temporary build with injected CUDA allocation failures exercised batch splitting and both mapped-host scratch paths. Independent and volume-matched WAVs matched the device-only outputs byte for byte. Cache corruption recovery, unavailable-cache compilation, and the tail and group audio regressions passed.

## Voice scheduling and scratch reuse

Compared with `2bdc204` on the RTX 4080 Laptop GPU, 128 jobs sampled from GigaMIDI and six SFZ libraries produced the same 89 accepted WAVs byte for byte and the same rejection reasons. Each binary ran twice with three passes per invocation. The median of the four warm passes fell from 3.402 s to 2.847 s, a 19.5% increase in accepted WAV throughput. Median CUDA event time fell from 3.158 s to 2.635 s, and peak explicit device allocations fell from 7.18 GiB to 6.95 GiB. A 32-job synthetic workload with mono, stereo, filtered, unfiltered, and pitch-modulated voices fell from 0.483 s to 0.379 s at the warm median.

With two separate 64-job processes per round and populated kernel caches, mean CUDA event time fell from 3.120 s to 2.592 s. Mean wall time rose from 8.786 s to 9.169 s over two rounds per binary. Device-wide utilization samples at or below 20% were 47 of 87 for the baseline and 47 of 91 for the updated renderer. These runs did not establish a fresh-process throughput or utilization improvement.

Injected CUDA allocation failures covered sample-bank growth in mapped host memory, batch splitting, and both scratch fallback paths. The WAVs matched normal device allocations byte for byte for independent jobs and volume-matched groups. Tail and group audio regressions passed.

## Batch budget and sample path resolution

Two isolated changes were compared against the renderer from the preceding section. The larger adaptive scratch budget increased peak allocations from 1.48 GiB to 2.66 GiB on the 32-job synthetic mix and reduced median warm wall time from 0.390 s to 0.344 s. All WAVs matched byte for byte. The sample-path lookup change reduced mean wall time from 9.089 s to 8.324 s on the 128-job, two-process workload, with the same accepted WAVs and rejection reasons. The larger CPU planning window took longer in the same process test.

The combined source built successfully. Windows application control denied execution of the combined binary (WinError 4551). The timings above describe the isolated changes. Combined performance and allocation-failure regression checks remain pending.

## Configurable batch job limits

The grouped audio regression renders a 66-job volume group with the default limits and with `--cpu-batch-jobs 1 --gpu-batch-jobs 128`. The latter completes the group in one CUDA batch without retaining or staging PCM between batches. The accepted WAVs match the default limits byte for byte.

## Preset audit

The audit uses the host-only ClaySampler loader and the sfizz C API on the same paths. It records accepted region counts, ClaySampler rejection reasons and unknown opcodes reported by sfizz. The result measures loading and sample resolution. Use the audio comparison harness for MIDI-specific behavior and sound.

```sh
python tests/audit_presets.py --clay build/claysampler --sfizz-library path/to/sfizz.dll \
  --root path/to/instruments --pattern '*/*/presets/*.sfz' --output build/sfz-audit.json
```
