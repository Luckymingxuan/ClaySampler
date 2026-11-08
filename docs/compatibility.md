# Compatibility

The first end-to-end presets were VSCO2 Clarinet Sustain and VCSL Grand Piano Steinway B. The list below describes accepted functionality. Individual combinations beyond the tested presets may still differ from sfizz.

| Area | Current coverage |
|---|---|
| SFZ input | Control, global, master, group and region headers, quoted includes, definitions, relative samples, `default_path`, `set_ccN` and `set_hdccN`. Display labels and images are accepted as metadata. |
| Sample mapping | Key and velocity ranges, pitch center, transpose, tuning, offset, end, forward loops with crossfades, and one-shot playback. Direction hints `loop_type` and `looptype` leave playback forward, following sfizz 1.2.3. |
| Triggers | Attack, first, legato, note release, release key, sustain pedal, CC ranges and CC-triggered samples, group choking, sequence counters, random ranges and last-note keyswitches. |
| Amplitude and pitch | Delay, attack, hold, decay, sustain and release, velocity tracking and per-region velocity curves, key and velocity crossfades, gain, pan, stereo width and position, and random gain and pitch. `ampeg_dynamic` is accepted with the implemented static envelope parameters. |
| MIDI | Format 0/1 with PPQ timing and a single channel per job. Notes, tempo changes, pitch bend, controller conditions, default CC7/10/11 automation, sustain CC64 and controller reset CC121. Program change has no effect when a preset is fixed by the job. |
| Audio | Mono and stereo samples via libsndfile, with stereo PCM16 WAV output. The current comparison uses WAV samples. |

Unsupported opcodes and headers fail the preset load. Unsupported MIDI messages fail the affected job. This includes SysEx, pressure, sostenuto and channel mode commands other than controller reset. MIDI CCs that have no mapping in the loaded preset or the supported default controller set have no effect.

Filters, effects, LFOs, bend smoothing, general CC modulation, additional keyswitch modes, release cancellation, voice stealing and custom `<curve>` blocks remain to be verified or implemented. The comparison CLI does not fall back to another renderer automatically.

## Preset load coverage

Each count is the number of presets that loaded and resolved their sample metadata. The scan used 375 local presets from six libraries. The load check measures parser and sample resolution coverage. Render checks use separate MIDI cases.

| Library | First renderer `aedaa4d` | Expanded support | Presets checked |
|---|---:|---:|---:|
| FreePats | 74 | 82 | 89 |
| Karoryfer | 2 | 28 | 129 |
| Meta Open Sound | 0 | 3 | 3 |
| SHLD | 2 | 11 | 11 |
| VCSL | 60 | 81 | 81 |
| VSCO2 | 54 | 62 | 62 |
| **Total** | **192** | **267** | **375** |

## Additional audio checks

FreePats Nylon Guitar, FreePats Spanish Classical Guitar and VSCO2 Solo Violin Vibrato passed the same 12-second phrase against sfizz in the initial implementation. The expanded implementation also passed the original 36-pair clarinet and piano suite.

The expanded implementation passed focused checks against sfizz for the FreePats Salamander and VCSL Knight piano pedal triggers, the SHLD Nebula Drift loop and release, Karoryfer Rubner Pizzicato Double Bass, and a VSCO2 Solo Violin Keyswitch phrase selecting the non-randomized articulations. Separate generated cases checked stereo width and position, velocity curves, crossfades, group choking, and a CC-only region with an initial controller value.

ClaySampler uses its own repeatable random sequence. The sampled VSCO2 keyswitch phrase with randomized articulation and Meta Open Sound steel guitar phrase with randomized pitch and gain both produced audio but exceeded the waveform threshold against sfizz. Five repeat renders with sfizz produced identical audio for each phrase. Exact matching for these random presets remains open.
