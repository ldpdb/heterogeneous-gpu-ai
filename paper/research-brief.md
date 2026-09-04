# Underused Fixed-Function GPU Engines as Auxiliary Accelerators for AI Workloads

## A research agenda for heterogeneous NVIDIA GPU inference and generation

> **Epistemic status:** This document presents research hypotheses rather than experimentally validated results. It is intended to encourage evaluation by researchers with expertise in GPU architecture, video processing, computer vision, and AI systems.

## Abstract

Modern NVIDIA GPUs are heterogeneous devices. In addition to CUDA/SM and Tensor cores, depending on architecture and product they may contain dedicated video encoders and decoders (NVENC/NVDEC), an Optical Flow Accelerator (OFA), RT hardware, copy engines, and platform-specific vision or inference accelerators.

AI discussions often focus on Tensor throughput, CUDA occupancy, memory capacity, and memory bandwidth. During some inference and generation workloads, however, specialized engines may be lightly used while programmable compute resources are heavily occupied. This motivates a narrower systems hypothesis: an AI pipeline may gain whole-device efficiency by reformulating useful auxiliary subproblems so that they resemble functions already implemented in specialized hardware.

The proposal is not to make NVENC, NVDEC, or OFA perform arbitrary matrix computation. It is to test mappings such as semantic segmentation plus optical-flow propagation, generative-video temporal constraints, codec-backed visual history, motion-derived side channels, and—more speculatively—fixed-function-assisted storage of non-video model state.

The strongest starting point is not a new claim: NVIDIA already documents an OFA-assisted object-tracking pipeline in which NVDEC decodes frames, a detector runs on the GPU, and the optical-flow engine helps track objects across frames. The open question is whether related mappings produce a measurable end-to-end benefit for modern AI workloads after representation, synchronization, memory, and quality costs are included.

## 1. Motivation and central question

Consider a simplified heterogeneous device:

| Resource | Documented primary role | Possible research role |
| --- | --- | --- |
| Tensor cores | Matrix operations and AI arithmetic | Primary inference/generation |
| CUDA/SM cores | Programmable GPU computation | Model kernels, conversion, refinement |
| OFA | Optical-flow estimation between images | Motion and correspondence signals |
| NVENC | Hardware video encoding and, on supported paths, motion estimation | Compressed history, motion statistics |
| NVDEC | Hardware video decoding | Retrieval of compressed visual history |
| Copy engines | Asynchronous data movement | Overlap of staging and model work |
| RT hardware | BVH traversal and ray/triangle intersection | Structured spatial-query side channel |
| Platform accelerators | Product-specific vision/image/inference operations | Platform-specific variants of the agenda |

The central question is:

> How much additional useful AI work can be extracted from a modern GPU by designing around the complete heterogeneous chip rather than only its programmable compute cores?

This question has a stricter form for each experiment:

> Does the heterogeneous pipeline improve whole-system output per unit time or energy at an acceptable quality cost, relative to a credible baseline, under a specified hardware and software configuration?

## 2. What is and is not being proposed

The agenda is an algorithmic-reformulation program:

`AI subproblem → compatible representation → fixed-function operation → auxiliary signal → model or system decision`

Examples include temporal coherence to optical flow, object persistence to mask warping, historical frames to encode/decode, and scene change to motion/residual statistics.

The agenda does not assert that:

- fixed-function engines are general-purpose processors;
- their advertised independence means zero contention or zero synchronization cost;
- a faster isolated engine automatically improves end-to-end inference;
- codec compression is better than learned or purpose-built latent/KV representations;
- RT hardware can execute arbitrary geometric or tensor algorithms;
- any proposed experiment has already succeeded.

## 3. Evidence taxonomy

Claims in this project should be read using the following levels:

### 3.1 Documented capability

NVIDIA documents dedicated optical-flow functionality, hardware video encode/decode APIs, NVENC motion-estimation-only mode on supported configurations, asynchronous data movement, and RT hardware for specific ray-tracing operations. These are hardware/API facts, not evidence of the proposed AI benefit.

### 3.2 Existing demonstrated use

NVIDIA’s NVOFA tracker combines NVDEC, GPU object detection, optical flow, and tracking logic. The relevant demonstrated pattern is co-use of a semantic detector with a dedicated motion engine. Research literature also demonstrates optical-flow-guided propagation and temporal consistency methods, though those papers do not establish that NVIDIA fixed-function OFA is the best implementation.

### 3.3 Plausible extension

Segmentation-mask propagation, motion-conditioned generative video, compressed visual memory, and NVENC motion estimation as a fallback or comparison are plausible extensions because their data dependencies resemble documented video operations. They need controlled measurements.

### 3.4 Highly speculative idea

Using a video codec as a non-video tensor or transformer KV-cache compressor is intentionally high-risk. The data layout, precision, codec restrictions, conversion overhead, and error propagation may make it inferior to purpose-built methods. A negative result would be informative.

## 4. Why optical flow is the first test

OFA is the clearest near-term case because its output—motion vectors between images—has a direct relationship to temporal correspondence. NVIDIA describes the accelerator as independent of CUDA cores and supplies an object-tracking example using a detector periodically and flow every frame. That establishes an implementation precedent, not a result for semantic segmentation or generative video.

### 4.1 Segmentation propagation

A candidate pipeline is:

1. Run a high-quality segmentation model on a key frame.
2. Propagate masks or features to later frames using OFA-derived flow.
3. Apply a small CUDA/SM refinement or confidence check.
4. Re-run full segmentation periodically or when confidence falls.

The model answers what a region is; flow helps estimate where it moved. The mapping is most plausible for regions with coherent motion, and weakest near occlusion boundaries, disocclusions, articulated motion, fast motion, blur, or large viewpoint changes.

The test must compare against per-frame segmentation and against any CUDA optical-flow or learned propagation baseline used by the project. Quality must be measured, not assumed.

### 4.2 Generative-video temporal consistency

Generated video often exhibits temporal drift in identity, texture, geometry, backgrounds, hands, or small details. A candidate system could use flow between generated or reference frames to produce motion-aware conditioning, confidence, or a consistency-control signal for the next denoising/generation step.

This does not replace generative inference. Occlusion, newly visible surfaces, camera changes, deformation, and interactions still require model reasoning. Flow may help provide a low-cost temporal cue, but it may also reinforce wrong correspondences or cause motion freezing. The first experiment should therefore be an ablation, with explicit quality and diversity metrics.

## 5. NVENC/NVDEC and compressed visual memory

NVENC/NVDEC expose a possible hardware-backed visual-history path:

`recent frames → NVENC → compressed history → VRAM/RAM/storage → NVDEC → reconstructed reference`

The motivation is not that codec compression is intrinsically superior. A raw frame history consumes memory and bandwidth; a codec path might reduce storage pressure while moving much of encode/decode work to dedicated engines. The relevant comparison is whole-system throughput, memory footprint, traffic, latency, energy, and reconstruction quality against raw frames and learned/latent history.

Application-allocated CUDA resources and reduced-copy paths may matter, but support and performance are SDK-, driver-, codec-, format-, and architecture-dependent. A pipeline that requires large CUDA conversions or host round trips may fail before reaching the engines.

## 6. NVENC motion estimation versus OFA

NVIDIA documents an NVENC motion-estimation-only mode that can return motion vectors and mode information on supported hardware/API paths. NVIDIA also recommends OFA for computer vision, AI, and frame interpolation because OFA vectors provide better visual matching for those use cases.

That recommendation makes a direct comparison useful rather than redundant. NVENC may still be valuable where block-level codec correspondence is sufficient, OFA is unavailable, codec decisions are useful, or a hybrid codec already exists. The experiment should compare quality, resolution, vector density, latency, fixed-function utilization, and interference with simultaneous encoding/AI work.

## 7. Structural and anomaly side channels

An auxiliary engine could observe coarse structural signals without duplicating the full model computation. Candidate signals include global or local motion, residual energy, motion-vector discontinuities, reference-frame changes, compressibility changes, and persistence of static regions.

These signals would be anomaly indicators, not semantic proof or mathematical error correction. A monitor might flag abrupt frame corruption, unexpected camera jumps, or coarse temporal discontinuities early enough to trigger regeneration. The research burden is to measure false positives, false negatives, detection latency, and whether recovery improves final quality rather than merely adding overhead.

RT hardware is an especially constrained variant. NVIDIA documents RT cores for BVH traversal and ray/triangle intersection, with application-managed ray generation and shading. A structural-query experiment would need a natural spatial representation, an OptiX or graphics API path, and evidence that acceleration-structure and launch costs do not overwhelm the query. It should not be described as using RT cores for arbitrary AI computation.

## 8. Non-video tensor and KV-cache compression

Arbitrary tensors could theoretically be laid out as image-like arrays and passed through a video codec. That does not make NVENC a general tensor compressor. Neural activations and transformer KV caches have statistics, precision requirements, layouts, and error sensitivities that differ from natural video.

The experiment is nevertheless worth isolating as a boundary test:

> Can mediocre compression on an otherwise-underused fixed-function engine compensate for inferior compression characteristics through concurrency, memory savings, or reduced SM work?

Candidate layouts might map feature dimension, attention head, and token position onto spatial or temporal axes. Comparisons must include uncompressed KV, a purpose-built KV quantization/compression method, and the fixed-function path. Evaluate perplexity or task quality, attention/output error, memory savings, encode/decode latency, token throughput, bandwidth, and GPU occupancy. This is a low-probability/high-information experiment, not a proposed replacement for modern KV-cache research.

## 9. Whole-device evaluation

The agenda can fail if it optimizes an isolated kernel or engine instead of useful system output. A CUDA implementation may be faster in isolation but harm a saturated inference pipeline; conversely, an OFA or NVENC operation may be slower in isolation but useful if it lets scarce SM resources remain on the model.

Every serious result should report:

- end-to-end throughput: frames/s, samples/s, or tokens/s;
- latency: average, p50, p95/p99, and synchronization stalls where relevant;
- Tensor-core and SM occupancy/utilization;
- fixed-function utilization and engine queueing;
- VRAM capacity and peak working set;
- device, host, and interconnect memory bandwidth/traffic;
- CPU utilization and host scheduling overhead;
- power, energy per frame/token, and clock/power policy;
- quality or accuracy, including temporal consistency and failure cases;
- preprocessing, format conversion, copies, encode/decode, and recovery overhead.

The desired quantity is useful whole-device output, not a claim that an auxiliary engine is universally faster.

## 10. Limitations and failure modes

Likely failure modes include:

- conversion overhead to NV12, YUV, image, or codec-compatible layouts;
- VRAM bandwidth and cache contention despite separate engine blocks;
- synchronization that serializes what appeared to be concurrent work;
- format, resolution, bit-depth, codec, API, driver, or platform restrictions;
- poor correspondence from optical flow under occlusion, blur, or deformation;
- codec artifacts or error propagation in visual/state memory;
- host-side API overhead or context switches;
- quality loss that exceeds the throughput or power benefit;
- a CUDA kernel or learned representation that is simply a better trade-off;
- insufficient instrumentation to establish actual overlap.

These are not footnotes. They define the falsification conditions for the agenda.

## 11. Research opportunity

AI workloads are increasingly multimodal, temporal, stateful, and memory-constrained. A future runtime might schedule semantic inference on Tensor/SM resources while using OFA for motion, NVENC/NVDEC for visual history, copy engines for staging, and product-specific accelerators for compatible image operations. That is a compiler/runtime and systems research opportunity only if the representations and dependencies can be made practical.

The modest claim is therefore:

> Some AI pipelines may leave useful silicon underused because their algorithms are designed around programmable compute resources rather than the full set of specialized processors on the device.

The answer may be “very little.” The purpose of this repository is to find out carefully.

## 12. Suggested first deliverables

1. A capability probe and environment report for one or more GPUs.
2. A minimal OFA-versus-CUDA flow microbenchmark with overlap instrumentation.
3. A segmentation propagation baseline with quality/failure analysis.
4. A raw-frame versus codec-backed visual-history benchmark.
5. A negative-result report for at least one non-video tensor layout.

See [EXPERIMENTS.md](../EXPERIMENTS.md) for experiment cards and acceptance criteria.
