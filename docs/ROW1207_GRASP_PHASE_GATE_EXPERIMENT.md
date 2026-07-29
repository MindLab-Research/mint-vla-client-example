# Row1207 grasp phase-gate experiment

## Purpose

This experiment tests whether row1207 fails primarily because the policy transitions to release/retract too early, or because the achieved multi-finger contact geometry cannot bear load.

The diagnostic gate is external to the policy. It does not modify language, state, model parameters, or raw policy predictions. `--phase-gate off` remains the default.

## Gate contract

`--phase-gate grasp-probe` observes live five-finger contact bits and object-floor support. When the configured finger count persists while floor support is clear, it:

1. latches the policy's contemporaneous finger servo target;
2. holds achieved root X/Y/orientation;
3. commands a small root-Z probe;
4. declares probe success only if the object follows, floor support remains clear, and multi-finger contact persists;
5. applies a larger retention lift only after probe success.

The evaluator writes the original model proposal, executed gate target, phase ID, override mask, and event evidence separately.

Example options:

```text
--chunk-stride 1
--phase-gate grasp-probe
--phase-gate-min-contact-count 4
--phase-gate-contact-persistence-frames 10
--phase-gate-probe-lift-mm 5
--phase-gate-probe-frames 20
--phase-gate-probe-follow-min-mm 3
--phase-gate-probe-min-contact-count 3
--phase-gate-retention-lift-mm 50
--phase-gate-retention-frames 100
--phase-gate-retention-follow-min-mm 20
--phase-gate-retention-min-contact-count 3
```

## Row1207 evidence

Row1207 is cube1/gesture09 and recorded-target replay Grade A. Its accepted80K Mode4 baseline contains57 three-finger frames and42 consecutive four-finger frames, but peak/final lift are6.54mm/-0.20mm.

A fresh independent action session produced zero three/four-finger frames. This revealed that separate stochastic sessions are not a valid causal A/B comparison. The accepted baseline's recorded servo targets were therefore replayed deterministically in MuJoCo. The replay reproduced contacts and physics flags exactly; maximum hand/object-position error was2.38e-7/7.45e-9.

The corrected gate triggered at local179 after four finger contacts and floor-clear persisted for10 frames. After a5mm/100ms root-Z probe:

```text
finger contacts:       4 (persisted)
object delta-Z:       -1.505mm
object-floor contact: true
probe:                failed
retention:            not entered
```

The gate increased the longest four-finger interval from42 to52 frames but did not produce lift: gated peak/final lift were4.85mm/-0.20mm. Visual inspection shows fingers on the top/near side without a clear opposing pinch.

## Conclusion

Premature phase transition is not sufficient to explain row1207. Four binary contact bits are not a secure-grasp predicate: the contact geometry remains non-load-bearing under a physical probe. The gate correctly rejects the grasp. Making the task succeed requires retry/reposition or stronger/opposing closure and a grasp-quality observation based on object follow/support, not a time progress scalar or contact count alone.

Generated evidence is intentionally Git-ignored at:

```text
results/inference/row1207_phase_gate_probe80k_20260729/
```
