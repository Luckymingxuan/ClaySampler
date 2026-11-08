# Renderer architecture

## Data flow

1. The SFZ loader resolves paths and inherited parameters, reads sample metadata, and indexes regions by key and triggering MIDI CC. MIDI files become frame-ordered event streams.
2. The CPU walks the sparse MIDI events, tracks keyswitches and controllers, resolves triggers and voice choking, and describes each active voice as one or more spans with source positions and envelope parameters. It builds compact automation tracks for MIDI controls, LFOs and supported flexible pitch envelopes. Sample PCM is decoded only for regions that actually trigger in the batch, then uploaded once.
3. Each CUDA block renders a tile of one output. Threads run 48-tap Kaiser-windowed sinc interpolation, blend loop crossfades, evaluate envelopes, and mix channels with the requested stereo width and position. Voices with variable pitch get a CUDA phase pass before interpolation. Filters and EQ keep causal state per voice in a CUDA pass after the parallel source rendering. These passes keep state across spans and share the batch memory budget. The final tile pass combines contributions and quantizes to PCM16.
4. The host retrieves the batch, checks for non-silent PCM, and writes WAV files. The batch API uses whole render jobs. The host does not advance the audio by synchronizing after each tile.

## Device behavior

NVRTC targets the selected GPU's architecture when supported by the installed compiler. For a newer GPU it emits PTX for the latest available compatible target. Launch sizing uses the CUDA occupancy API. Batches use a budget based on currently available VRAM, with a 512 MiB cap on batch staging. The decoded sample bank stays on the device across repeated passes. The CLI reports peak explicit CUDA allocations and CUDA event timing separately from end-to-end timing.

One process renders one SFZ preset at a time. The whole set of samples referenced by a batch must fit in VRAM. Large-instrument paging, asynchronous writeback, and multi-device scheduling are outside the current implementation. The code is device-adaptive, with performance verified so far on one device.

## Upstream references

ClaySampler uses its own parser, MIDI event planner, batch layout, and CUDA implementation. We used sfizz 1.2.3 as the reference for observable behavior, including its windowed-sinc approach, default CC mappings and master gain, envelope curves, and bend interpolation. The implementation does not include sfizz source or link sfizz at runtime. The comparison harness invokes the reference CLI separately. libsndfile is loaded as an external runtime dependency.

- [sfizz 1.2.3](https://github.com/sfztools/sfizz/releases/tag/1.2.3)
- [sfizz 1.2.3 BSD-2-Clause license](https://github.com/sfztools/sfizz/blob/1.2.3/LICENSE)
- [NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)
