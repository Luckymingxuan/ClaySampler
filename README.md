# ClaySampler

ClaySampler renders batches of MIDI files through SFZ instruments on NVIDIA CUDA GPUs. It targets offline audio generation, with sfizz 1.2.3 as the compatibility and quality reference.

The current implementation covers a tested SFZ subset. See [compatibility](docs/compatibility.md) for accepted features and [validation](docs/validation.md) for the sfizz benchmark and listening report.

The Windows workflow publishes x64 ZIPs and SHA-256 files as [GitHub Release assets](https://github.com/Luckymingxuan/ClaySampler/releases). See the [release guide](docs/releasing.md) for packaging and verification.

## Build and render

Build with CMake 3.20 or newer and a C++17 compiler. At runtime the executable needs an NVIDIA driver, NVRTC with its matching builtins library, and libsndfile. The executable contains the CUDA source and compiles it for the selected GPU. The validation scripts use Python.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Run the built `claysampler` executable. Pass as many MIDI paths as needed for the same preset. Output files use the MIDI basename.

```sh
claysampler --sfz instrument.sfz --output renders take1.mid take2.mid
```

Use `--nvrtc` and `--sndfile` for explicit library paths, or set `CLAY_NVRTC` and `CLAY_SNDFILE`. The Windows defaults are `nvrtc64_120_0.dll` and `libsndfile-1.dll`. On Linux they are `libnvrtc.so` and `libsndfile.so.1`. Use `--device` for a different CUDA ordinal and `--samplerate` to change the default of 44,100 Hz.

Successful jobs print `OK` and write PCM16 WAVs. Rejected MIDI jobs print `REJECT` with the input and reason. A batch containing rejected jobs returns exit status 2. A fatal setup error returns exit status 1.

During batch rendering, the CPU plans the next group and writes completed WAVs while the GPU renders and finishes the current group. In `natural` mode, the GPU sends the finished mono PCM16 samples and output metadata to the host for WAV writing.

To check whether a preset loads and its sample headers resolve, run `claysampler --check-sfz preset.sfz`. The load check runs without a CUDA device or MIDI input. Place `--sndfile path` before the preset paths when an explicit decoder library is needed.

## Tail policies

- `--tail-policy natural` is the default and produces mono WAVs using ClayEngine's `sfizz_native_mono_v7` finishing rules. It releases held pedals at the end of the performance, detects silence in the floating-point mix, and shortens an extended quiet tail with a gentle fade. `--max-audio-seconds` optionally bounds the entire render. A job that hits the limit before becoming quiet is rejected.
- `--tail-policy orig` produces stereo WAVs with the previous ClaySampler release and file-length behavior.
- `--tail-policy custom --tail-max-seconds 2 --tail-fade-ms 100` produces stereo WAVs with up to two seconds after the last MIDI performance or controller event. `--tail-max-seconds` is required and accepts fractional seconds from 0 to 3600. Zero ends the output at the last event. `--tail-fade-ms` accepts whole milliseconds from 0 to 3600000, defaults to 100, and can be set to zero to disable the fade. The fade applies only when the limit shortens the tail and stays within the tail.

[Tail policy details and checks](docs/validation.md#tail-policies)

## GPU quality gate and group volume

Both options are off by default and can be enabled independently. The quality gate runs on the finished PCM before any group gain is applied. A rejected render returns the reason and available statistics without downloading its PCM. MIDI eligibility, retries and replacement jobs are handled by the caller.

```sh
claysampler --sfz instrument.sfz --output renders --quality-gate clayengine take1.mid
claysampler --jobs tasks.tsv --group-volume mean --quality-gate clayengine
```

The UTF-8 job manifest has no header. Put the group ID, SFZ path, MIDI path and output WAV path on each row, separated by tabs. Relative paths are resolved against the manifest directory. Use `-` for an independent render. The caller assigns group IDs, and members can appear anywhere in the manifest.

```tsv
take-01	piano.sfz	take1.mid	renders/take1-piano.wav
take-01	harp.sfz	take1.mid	renders/take1-harp.wav
-	piano.sfz	take2.mid	renders/take2-piano.wav
```

With `--group-volume mean`, the GPU averages the body RMS of the accepted members and applies one fixed gain to each track. The body ends at the last matched note-off. The common target is lowered if needed to keep every peak at or below −1 dBFS. `--group-min-members` sets the minimum number of usable members and defaults to 2. If a group cannot reach the minimum, its surviving tracks are rejected with `insufficient_qualified_group_members`. `--group-wait-mib` caps device PCM retained across render batches and defaults to 256 MiB. A setting of zero uses host staging for groups that span batches.

`--quality-gate clayengine` applies the defaults below. `--quality-gate custom` uses the same defaults and accepts any of these overrides. `--quality-gate off` is the default.

| Override | ClayEngine preset |
| --- | ---: |
| `--qa-max-seconds` | 375 s, including the tail |
| `--qa-min-onset-median-dbfs` | −38 dBFS |
| `--qa-min-onset-quartile-dbfs` | −45 dBFS |
| `--qa-silence-dbfs` | −70 dBFS |
| `--qa-max-silence-fraction` | 0.75 |
| `--qa-max-silence-seconds` | 10 s of consecutive body silence |
| `--qa-min-peak` | 32 PCM16 units |
| `--qa-max-clipped-fraction` | 0.0001 |

In stereo modes the gate measures body and onset levels on the channel average, and checks clipping and peak across the actual output channels. `QUALITY` rows report the output path, pre-gain verdict, frame count, onset median and lower quartile (dBFS), silent fraction, longest silence (seconds) and clipped fraction. `VOLUME` rows report the path, original body RMS, target RMS (PCM16 units) and applied gain (dB). With either feature enabled, `TRANSFER` rows report the pass, final PCM bytes downloaded, intermediate PCM bytes staged to the host and peak retained device PCM bytes so far.

[Grouped audio validation and ClayEngine comparison](docs/validation.md#gpu-quality-and-group-volume)

[Architecture and upstream references](docs/architecture.md)
