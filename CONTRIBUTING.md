# Contributing

Contributions are welcome from researchers and engineers who want to benchmark, disprove, refine, or implement any part of this agenda. A negative result, unsupported-device report, or careful measurement of overhead is as valuable as a speedup.

## Good first contributions

- Add a capability probe for a documented GPU/SDK path.
- Reproduce the NVIDIA OFA tracker pattern and measure the resource timeline.
- Add a segmentation-propagation baseline with public data.
- Compare OFA and NVENC motion information on a declared workload.
- Benchmark raw versus compressed visual history.
- Document an API, format, driver, or architecture limitation.
- Add a published baseline for KV-cache or video temporal consistency.

## Before coding

Open an issue or draft note that states:

1. the research question and falsifiable hypothesis;
2. the proposed hardware/API path;
3. the baseline and ablations;
4. the expected quality and systems metrics;
5. the conditions that would count as a negative result.

Keep the initial implementation narrow. This project does not require one universal framework: CUDA/C++, Python bindings, the Optical Flow SDK, Video Codec SDK, PyNvVideoCodec, DeepStream, OptiX, or another documented interface may be appropriate for different experiments.

## Reproducibility expectations

Record GPU model and architecture, driver, OS, CUDA/SDK/API versions, clock and power policy, model/checkpoint, precision, input format/resolution, dataset/license, build commands, synchronization policy, and benchmark commit.

Report both isolated and end-to-end behavior. Include end-to-end throughput, Tensor/SM occupancy, fixed-function utilization, VRAM, bandwidth, latency, CPU use, power/energy, and quality/accuracy. Include warm-up, repetitions, percentiles, and failure cases.

Do not call a pipeline “concurrent” without timeline evidence. Do not call an engine “free” when it consumes bandwidth, copies, synchronization, power, or host work.

## Evidence labels

Use these labels in pull requests and documentation:

- **Documented capability**
- **Demonstrated use**
- **Plausible extension**
- **Highly speculative**
- **Measured result** (only with artifacts and environment details)

Update [references.md](references.md) for substantive technical claims. Keep vendor claims, related research, and local measurements visibly separate.

## Data and artifacts

Use public datasets or obtain permission. Do not commit credentials, proprietary media, model weights, large generated files, or profiler captures containing sensitive information. Prefer small summaries in Git and external artifact links with checksums when needed.

## Review standard

Reviews should ask:

- Is the claim stronger than the evidence?
- Is the baseline credible and fairly configured?
- Did conversion, synchronization, copies, and memory traffic get measured?
- Does quality or accuracy remain acceptable?
- Could another researcher reproduce or challenge the result?

Please be constructive. The purpose of this repository is to learn where heterogeneous GPU scheduling helps and where it does not.
