# Heterogeneous GPU AI

## Underused Fixed-Function GPU Engines as Auxiliary Accelerators for AI Workloads

This repository is an open research agenda for testing whether otherwise-underused fixed-function hardware on NVIDIA GPUs can perform useful auxiliary work while Tensor cores and CUDA/SM resources are occupied by AI inference or generation.

The agenda considers the Optical Flow Accelerator (OFA), NVENC, NVDEC, copy engines, RT hardware, and related platform accelerators. The central question is systems-oriented:

> Can an AI application produce more useful output from the whole GPU by mapping compatible subproblems onto specialized engines, even when those engines cannot execute arbitrary neural-network operations?

This is not a completed invention, implementation, or experimentally validated technique. The repository is designed so researchers can benchmark, disprove, refine, or implement any of the hypotheses.

## Evidence levels

The documents use four labels:

- **Documented capability** — described in authoritative NVIDIA documentation or an API specification.
- **Demonstrated use** — an existing NVIDIA sample, SDK application, or peer-reviewed/published system demonstrates the relevant pattern.
- **Plausible extension** — a reasoned mapping that remains to be tested in the proposed AI workload.
- **Speculative idea** — a high-risk concept whose representation, API path, or usefulness is uncertain.

The presence of a hardware capability does not imply that an AI workload will benefit. Conversion, synchronization, memory bandwidth, quality loss, and contention can erase any advantage.

## Start here

- [Research brief](paper/research-brief.md) — motivation, scope, evidence levels, hypotheses, limitations, and open questions.
- [Experiments](EXPERIMENTS.md) — six proposed experiment families, ordered by expected information value and increasing speculation.
- [Hardware notes](docs/hardware.md) — capability and API notes with platform caveats.
- [References](references.md) — NVIDIA documentation and relevant research.
- [Contributing](CONTRIBUTING.md) — how to add measurements, negative results, and reproducible implementations.

## Initial research sequence

1. OFA-assisted segmentation propagation.
2. OFA-assisted generative-video temporal consistency.
3. NVENC/NVDEC compressed visual memory.
4. NVENC motion estimation versus OFA.
5. Fixed-function structural/anomaly side channels.
6. Experimental non-video tensor/KV compression.

## What would count as evidence?

At minimum, compare a proposed heterogeneous pipeline against a credible CUDA/SM-only or existing-system baseline. Report end-to-end throughput, latency, Tensor/SM occupancy, fixed-function utilization, VRAM use, memory bandwidth, power or energy, and quality/accuracy. Include warm-up policy, synchronization behavior, hardware/driver/SDK versions, and failure cases.

## Status

This is an initial research landing zone. No experiment in this repository should be read as having succeeded until reproducible results are added.

## License

Unless otherwise noted, repository material is available under the [Apache License 2.0](LICENSE).
