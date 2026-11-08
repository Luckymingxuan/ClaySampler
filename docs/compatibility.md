# Compatibility

The first end-to-end presets are VSCO2 Clarinet Sustain and VCSL Grand Piano Steinway B. See the generated A/B report for the exact tested inputs. The list below describes accepted functionality. Individual combinations beyond the tested presets may still differ from sfizz.

| Area | Current coverage |
|---|---|
| SFZ input | Control, global, master, group and region headers, quoted includes, definitions, relative samples, `default_path` and `set_ccN`. |
| Sample mapping | Key and velocity ranges, pitch center, transpose, tuning, offset, end, forward loops, and one-shot playback. |
| Triggers | Attack, first, legato, note release, release key, sustain pedal, CC ranges, sequence counters and random ranges. ClaySampler uses its own repeatable random sequence. |
| Amplitude | Delay, attack, hold, decay, sustain and release, velocity tracking, gain and pan. `ampeg_dynamic` is accepted with the implemented static envelope parameters. |
| MIDI | Format 0/1 with PPQ timing and a single channel per job. Notes, tempo changes, pitch bend, controller conditions, default CC7/10/11 automation, sustain CC64 and controller reset CC121. Program change has no effect when a preset is fixed by the job. |
| Audio | Mono and stereo samples via libsndfile, with stereo PCM16 WAV output. The current comparison uses WAV samples. |

Unsupported opcodes and headers fail the preset load. Unsupported MIDI messages fail the affected job. This includes SysEx, pressure, sostenuto and channel mode commands other than controller reset. MIDI CCs that have no mapping in the loaded preset or the supported default controller set have no effect.

Filters, effects, LFOs, keyswitches, complex CC triggers, release cancellation, voice stealing, custom curves and advanced sample behaviors remain to be verified or implemented. The comparison CLI does not fall back to another renderer automatically.

## Additional preset checks

The same 12-second MIDI phrase passed the automatic sfizz comparison on FreePats Nylon Guitar, FreePats Spanish Classical Guitar and VSCO2 Solo Violin Vibrato. Relative waveform error ranged from 0.89% to 1.16% after alignment.

Five other presets stopped at the following first unsupported opcode. A preset can contain more unsupported opcodes after the first one.

| Preset | First unsupported opcode |
|---|---|
| Meta Open Sound Steel Guitar | `amp_random` |
| VSCO2 Solo Violin Keyswitch | `sw_default` |
| SHLD Nebula Drift | `loop_crossfade` |
| Karoryfer Bigcat Plucked Cello | `label_cc7` |
| Karoryfer Rubner Pizzicato Double Bass | `amp_velcurve_54` |
