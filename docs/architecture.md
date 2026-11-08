# Renderer architecture

## Data flow

1. The SFZ loader resolves paths and inherited parameters, reads sample metadata, and indexes regions by key and triggering MIDI CC. MIDI files become frame-ordered event streams.
2. The CPU walks the sparse MIDI events, tracks keyswitches and controllers, resolves triggers and voice choking, and describes each active voice as one or more spans with source positions and envelope parameters. It builds compact automation tracks for MIDI controls, LFOs and supported flexible pitch envelopes. Sample PCM is decoded only for regions that actually trigger in the batch, then uploaded once.
3. Each CUDA block renders a tile of one output. Threads run 48-tap Kaiser-windowed sinc interpolation, blend loop crossfades, evaluate envelopes, and mix channels with the requested stereo width and position. Voices with variable pitch get a CUDA phase pass before interpolation. Filters and EQ keep causal state per voice in a CUDA pass after the parallel source rendering. These passes keep state across spans and share the batch memory budget.
4. CUDA analyzes the float mix for `natural`, applies ClayEngine's mono tail finishing rules, and packs the accepted PCM16 frames. `orig` quantizes directly to stereo PCM16. `custom` bounds voice spans in the planner and applies any fade on CUDA. The GPU also records each output's peak and audible extent. The host downloads the final PCM and metadata, then writes the WAV, including any logical zero padding beyond the last planned frame.

The CPU prepares the first group while CUDA initializes and plans the next group while CUDA renders the current group. WAV workers write earlier groups while both stages proceed. The renderer uploads only newly decoded samples at the group boundary, keeping their offsets stable across the invocation. It reuses device scratch space and up to three page-locked host PCM buffers. The bounded queues keep memory use predictable as the input list grows.

## Device behavior

NVRTC targets the selected GPU's architecture when supported by the installed compiler. For a newer GPU it emits PTX for the latest available compatible target. NVRTC 12.4 and newer use four compiler optimization threads when the host has at least four hardware threads. Launch sizing uses the CUDA occupancy API. Batches use a budget based on total and available VRAM. The decoded sample bank stays on the device across repeated passes. The CLI reports peak explicit CUDA allocations and CUDA event timing separately from end-to-end timing.

One process renders one SFZ preset at a time. Its decoded sample bank and current batch scratch must fit in VRAM. Performance measurements use one device.

## Upstream references

ClaySampler uses its own parser, MIDI event planner, batch layout, and CUDA implementation. We used sfizz 1.2.3 as the reference for observable behavior, including its windowed-sinc approach, default CC mappings and master gain, envelope curves, and bend interpolation. The implementation does not include sfizz source or link sfizz at runtime. The comparison harness invokes the reference CLI separately. libsndfile is loaded as an external runtime dependency.

ClayEngine's `instruments/midi_render.py` and `instruments/render_tail.py` are the behavioral references for `natural`. ClaySampler prepares the events in memory and implements the float stop criterion and PCM16 mono finishing on CUDA.

- [sfizz 1.2.3](https://github.com/sfztools/sfizz/releases/tag/1.2.3)
- [sfizz 1.2.3 BSD-2-Clause license](https://github.com/sfztools/sfizz/blob/1.2.3/LICENSE)
- [NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)
