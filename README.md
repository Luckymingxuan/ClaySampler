# ClaySampler

ClaySampler renders batches of MIDI files through an SFZ instrument on NVIDIA CUDA GPUs. It targets offline audio generation, with sfizz 1.2.3 as the compatibility and quality reference.

The current implementation covers a tested SFZ subset. See [compatibility](docs/compatibility.md) for accepted features and [validation](docs/validation.md) for the sfizz benchmark and listening report.

## Build and render

Build with CMake 3.20 or newer and a C++17 compiler. At runtime the executable needs an NVIDIA driver, NVRTC with its matching builtins library, and libsndfile. The executable contains the CUDA source and compiles it for the selected GPU. Python is used only by the comparison harness.

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

[Architecture and upstream references](docs/architecture.md)
