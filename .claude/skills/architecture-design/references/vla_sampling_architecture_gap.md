# VLA Sampling Architecture Gap

Date: 2026-04-04

This note documents the current OpenPI sampling architecture in PR 422 and why it is still a mismatch with the intended MinT design.

## Current behavior

Training side:

- one shared OpenPI trainer actor per base model
- per-tenant state kept as session state
- tenant switching handled by saving and loading session state on the shared trainer actor

Sampling side:

- `save_weights_for_sampler` materializes an inference-ready checkpoint for a single tenant
- `create_action_session` starts a dedicated OpenPI action actor for that checkpoint
- one action session therefore implies one action actor and one full sampler checkpoint load
- concurrent sampling across tenants scales by creating more action actors, subject to GPU capacity

## Why this is a MinT architecture mismatch

MinT is intended to separate training and sampling while still treating sampling as a shared serving substrate.

The current OpenPI sampling path does not do that.

It achieves isolation by:

- exporting a full inference checkpoint per session
- loading that checkpoint into a dedicated action actor
- keeping tenant isolation at the actor boundary

That means sampling multi-tenancy is not implemented as:

- a shared sampler actor keyed by base model
- session switching inside one sampler runtime
- or a shared adapter-style serving surface

It is implemented as checkpoint-per-session actor spawning.

## Evidence from runtime behavior

- pressure on the sampling side shows up as action-actor GPU pressure
- action-session creation can fail when no additional action actor can be placed on the assigned worker
- trainer sharing works, but sampling does not share the same way

This matches the observed architecture, not just an abstract complaint.

## Minimal design change needed

A MinT-clean OpenPI sampling design would need all of the following:

1. shared action runtime per base model
- one action runtime actor per base model, not per action session

2. per-session checkpoint switching inside that runtime
- action sessions must map to checkpoint identity or session state without requiring a new Ray actor per tenant

3. worker protocol support for switching weights
- the current action workers only support `create_session`, `act`, and `shutdown`
- they do not support `load_weights` or `load_session_state`
- that is the protocol gap that prevents a shared action runtime from multiplexing tenants

4. action session manager routing change
- `action_session_manager` would need to route many action sessions through one shared runtime keyed by base model, analogous to how the training side routes many training sessions through one shared trainer actor

5. capacity semantics change
- capacity management would need to treat action sessions as logical sessions within a shared sampler, not as one-GPU actor placements

## Practical implication

The current implementation can be made operational and tested, but it should not be mistaken for the final MinT architecture.

The right status is:

- training architecture: mostly aligned
- sampling architecture: still a known structural gap

That gap should stay explicit in validation and reporting.
