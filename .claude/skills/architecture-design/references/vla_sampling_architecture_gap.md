# VLA Sampling Architecture Gap

Date: 2026-04-07

This historical note documents the OpenPI sampling architecture observed in PR
422 on 2026-04-07 and why it was still a mismatch with the intended MinT design.

## Observed behavior on 2026-04-07

Training side:

- one shared OpenPI trainer actor per base model
- per-tenant state kept as session state
- tenant switching handled by saving and loading session state on the shared trainer actor

Sampling side:

- `save_weights_for_sampler` materializes an inference-ready checkpoint for a single tenant
- `create_action_session` now routes through one shared OpenPI action runtime per base model
- tenant switching inside that shared runtime is handled by saving and loading action-session state on vePFS
- concurrent sampling across tenants no longer implies one Ray actor per action session in the tested mixed-case path
- but tenant isolation is still achieved by checkpoint-per-session state materialization, not by a true shared adapter-style serving substrate

## Why this is a MinT architecture mismatch

MinT is intended to separate training and sampling while still treating sampling as a shared serving substrate.

The OpenPI sampling path observed in this run did not do that.

It achieves isolation by:

- exporting a full inference checkpoint per session
- saving checkpoint-derived action-session state per tenant on vePFS
- reloading that session state inside one shared action actor

That means sampling multi-tenancy is only partially implemented as:

- a shared sampler actor keyed by base model: yes, now present in the tested path
- session switching inside one sampler runtime: yes, now present in the tested path
- a shared adapter-style serving surface: still no

The remaining gap is that the shared sampler still depends on checkpoint-per-session state materialization rather than a lighter-weight shared serving substrate.

## Evidence from runtime behavior

- the mixed valid/invalid isolation probe now reuses one shared action actor while preserving the tested session boundary
- action-session contamination was previously real and was fixed by moving action-session state roots from checkpoint-scoped paths to actor-scoped vePFS paths
- trainer sharing works, and sampling now shares structurally too, but it still reloads checkpoint-derived tenant state instead of using a true shared adapter surface

This matches the observed architecture, not just an abstract complaint.

## Minimal design change needed

A MinT-clean OpenPI sampling design would still need all of the following:

1. keep the shared action runtime per base model
- this part now exists and should remain

2. lighter-weight per-session switching inside that runtime
- action sessions should switch lightweight tenant state without full checkpoint-derived state materialization

3. serving abstraction that is not checkpoint-per-session
- the current shared runtime multiplexes tenants by persisting and restoring checkpoint-derived action state
- it does not yet behave like MinT shared adapter serving

4. action session manager routing must stay aligned with the shared runtime
- this routing now exists for the tested path and must not regress

5. capacity semantics that reflect logical sessions rather than checkpoint-derived sampler state churn
- observed capacity pressure was lower than actor-per-session spawning, but sampling still paid checkpoint-derived state-switch costs

## Practical implication

The implementation observed in this run could be made operational and tested,
but it should not be mistaken for the final MinT architecture.

The right status is:

- training architecture: mostly aligned
- sampling architecture: improved substantially
- sampling serving model: still not fully MinT-clean

That gap should stay explicit in validation and reporting.
