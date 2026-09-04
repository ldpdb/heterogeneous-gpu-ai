# AGENTS.md

## Project purpose

This repository is an open research agenda, benchmark plan, and landing zone for studying whether underused fixed-function NVIDIA GPU engines can provide useful auxiliary work during AI inference or generation.

It is not a claim that a new accelerator, algorithm, or performance result has been invented. Treat every unmeasured proposal as a hypothesis.

## Research and writing rules

- Preserve epistemic labels. Distinguish **documented capability**, **demonstrated use**, **plausible extension**, and **speculative idea**.
- Cite primary sources whenever describing NVIDIA hardware, APIs, SDK behavior, or published research.
- Never turn a vendor capability statement into a claim of end-to-end AI benefit.
- Never report an experiment as successful unless the repository contains reproducible artifacts, environment details, raw measurements, and an honest comparison baseline.
- Make negative results welcome. A careful disproof, limitation, or architectural boundary is a valuable contribution.
- Keep claims scoped to GPU architecture, driver, SDK, API, resolution, codec, model, and workload conditions. Hardware support varies by generation and platform.
- Separate isolated engine throughput from whole-device and end-to-end performance.
- Report synchronization, format conversion, memory movement, and scheduling overhead; these are part of the proposed technique.
- Do not use proprietary or private data in examples or benchmarks without permission. Prefer public datasets and document licenses.
- Do not commit model weights, large media, credentials, API keys, profiler dumps containing sensitive paths, or generated build directories.

## Experiment expectations

Every experiment or implementation should document:

1. Research question and falsifiable hypothesis.
2. Exact hardware, driver, CUDA/SDK versions, OS, power mode, and clock policy.
3. Baseline and ablations, including a CUDA/SM-only baseline where applicable.
4. Data representation, resolution, batch/sequence settings, and quality metric.
5. Timeline or synchronization evidence showing whether overlap actually occurred.
6. End-to-end results plus per-engine utilization and resource costs.
7. Failure cases, quality regressions, and conditions under which the idea should be rejected.

## Contribution hygiene

- Prefer small, reviewable changes.
- Update `references.md` when adding a substantive technical claim.
- Add or update an experiment card in `EXPERIMENTS.md` before adding benchmark code.
- Avoid naming a result “accelerated,” “free,” “parallel,” or “real time” without measurements supporting that wording.
- Use neutral language in issues and reviews. The goal is to benchmark, refine, or disprove the agenda.

## Current implementation stance

The initial repository is intentionally implementation-neutral. Contributors may use the NVIDIA Optical Flow SDK, Video Codec SDK, PyNvVideoCodec, CUDA/C++, Python bindings, DeepStream, OptiX, or another documented interface when it is the right tool for a particular experiment. Record the choice and its constraints rather than making it a project-wide dependency prematurely.
