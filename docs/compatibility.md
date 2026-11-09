# Compatibility

The first end-to-end presets were VSCO2 Clarinet Sustain and VCSL Grand Piano Steinway B. The list below describes accepted functionality. Individual combinations beyond the tested presets may still differ from sfizz.

| Area | Current coverage |
|---|---|
| SFZ input | Control, global, master, group, region and linear `<curve>` headers, quoted includes, definitions, relative samples, `default_path`, `set_ccN` and `set_hdccN`. Display labels and images are accepted as metadata. |
| Sample mapping | Key and velocity ranges, pitch center, transpose, tuning, fixed, CC and random offsets, end, forward loops with crossfades, one-shot playback, and sine, silence and uniform noise generators. Direction hints `loop_type` and `looptype` leave playback forward, following sfizz 1.2.3. |
| Triggers | Attack, first, legato, note release, release key, sustain pedal, CC ranges and CC-triggered samples, group choking, per-note polyphony, sequence counters, random ranges, and last, held, released and previous-note keyswitches. |
| Amplitude and pitch | ADSR with velocity and CC modulation at trigger time, per-region velocity curves, key, velocity and CC crossfades, controller gain, pan, stereo width and position, random gain and pitch, and pitch modulation. The flexible pitch envelope supports linear points ending at zero with no separate release segment. |
| Filters and modulation | One or two low-pass, high-pass, band-pass or notch filters from the implemented types, filter envelopes, three peaking EQ bands, pitch envelopes, and common legacy and numbered LFO waveforms and destinations. |
| MIDI | Format 0/1 with PPQ timing and a single channel per job. Notes, tempo changes, pitch bend and bend smoothing, controller conditions, default CC7/10/11 automation, sustain CC64 and controller reset CC121. `natural` also handles default CC66 sostenuto and CC120/123. Program change has no effect when a preset is fixed by the job. |
| Audio | Mono and stereo samples via libsndfile. PCM16 WAV output is mono in `natural` and stereo in `orig` and `custom`. The current comparison uses WAV samples. |

Unsupported opcodes and headers fail the preset load. Unsupported MIDI messages fail the affected job. This includes SysEx and pressure, as well as sostenuto and channel mode commands other than controller reset in `orig` and `custom`. MIDI CCs that have no mapping in the loaded preset or the supported default controller set have no effect.

ClaySampler accepts a limited set of spellings that sfizz 1.2.3 ignores, including the tested `varN_` forms, inter-LFO frequency CC spellings and envelope shape hints. Other unsupported directives still reject the preset. Stateful effect buses, additional oscillators and filter types, general flexible-envelope routing, release cancellation, and global voice stealing remain open. A job that changes an unsupported dynamic envelope stage mid-voice reports that limitation. The CLI does not fall back to another renderer automatically.

## Preset load coverage

Each count is the number of presets that loaded and resolved their sample metadata. The audit ran both engines on the same 375 local presets from six libraries. Every preset loaded with the same region count in ClaySampler and sfizz 1.2.3. The sfizz API reported unknown opcodes in 112 presets. Render checks use separate MIDI cases.

| Library | ClaySampler `5daa82c` | Current ClaySampler | sfizz 1.2.3 |
|---|---:|---:|---:|
| FreePats | 82 | 89 | 89 |
| Karoryfer | 28 | 129 | 129 |
| Meta Open Sound | 3 | 3 | 3 |
| SHLD | 11 | 11 | 11 |
| VCSL | 81 | 81 | 81 |
| VSCO2 | 62 | 62 | 62 |
| **Total** | **267** | **375** | **375** |

## Additional audio checks

FreePats Nylon Guitar, FreePats Spanish Classical Guitar and VSCO2 Solo Violin Vibrato passed the same 12-second phrase against sfizz in the initial implementation. The expanded implementation also passed the original 36-pair clarinet and piano suite.

The expanded implementation passed focused checks against sfizz for the FreePats Salamander and VCSL Knight piano pedal triggers, the SHLD Nebula Drift loop and release, Karoryfer Rubner Pizzicato Double Bass, and a VSCO2 Solo Violin Keyswitch phrase selecting the non-randomized articulations. Separate generated cases checked stereo width and position, velocity curves, crossfades, group choking, and a CC-only region with an initial controller value.

ClaySampler uses its own repeatable random sequence. The sampled VSCO2 keyswitch phrase with randomized articulation and Meta Open Sound steel guitar phrase with randomized pitch and gain both produced audio but exceeded the waveform threshold against sfizz. Five repeat renders with sfizz produced identical audio for each phrase. Exact matching for these random presets remains open.

The current implementation passed the original 36-pair suite again and 15 of 19 new comparisons. Passing generated cases covered CC and envelope modulation, filter cascades, LFO pitch, pitch and filter envelopes, sine generation and EQ. Passing real presets included FreePats Piano and Tenor Saxophone, Karoryfer Bigcat Plucked Cello, Black Twang Guitar, SneakyBass Pluck and Meatbass Arco. The four cases below produced audio and exceeded the automatic waveform threshold of 18%.

| Case | Relative waveform error |
|---|---:|
| Bend smoothing with pitch CC | 25.63% |
| Linear flexible pitch envelope | 28.21% |
| Cowsynth Asthmatic Pad with random offsets | 140.87% |
| Bear Sax Solo Mono with randomized regions and offsets | 110.54% |

These results establish load coverage for the audited presets and audio agreement for the passing MIDI cases. The loader applies the same rules to SFZ files from any library. Use `claysampler --check-sfz preset.sfz` to check a new preset and the [audio comparison harness](validation.md) to compare its rendering with sfizz. Other MIDI control combinations and rendering performance on the new DSP paths need separate validation.
