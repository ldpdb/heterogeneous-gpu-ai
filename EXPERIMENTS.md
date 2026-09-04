# Experiments

This file is a queue of falsifiable experiments, not a list of claimed results. Each card should be implemented with a reproducible baseline, instrumentation, and an explicit decision rule.

## Shared benchmark protocol

For every experiment, record:

- GPU model, architecture, VRAM, driver, OS, CUDA/SDK version, power mode, and clock policy;
- model name/checkpoint, precision, batch or sequence settings, input resolution, codec, pixel format, and dataset/license;
- warm-up count, repetitions, synchronization points, stream/event configuration, and whether host round trips occur;
- baseline, heterogeneous variant, and ablations that remove the fixed-function stage or conversion/copy stage;
- end-to-end throughput, latency distribution, Tensor/SM occupancy, fixed-function utilization, VRAM use, bandwidth, CPU use, power/energy, and quality/accuracy;
- raw logs or machine-readable summaries plus failure cases.

“Concurrent” must be demonstrated with a timeline or equivalent evidence. A host API returning early is not proof that engines overlapped.

## 1. OFA-assisted segmentation propagation

**Evidence level:** Plausible extension grounded in documented OFA capability and NVIDIA’s demonstrated object-tracking pattern.

**Question:** Can high-quality segmentation on key frames plus OFA-based mask propagation reduce end-to-end cost while preserving acceptable segmentation quality?

**Hypothesis:** On videos with coherent motion and moderate scene change, propagating masks between key frames may reduce SM/Tensor work enough to improve whole-system throughput or energy per frame.

**Candidate design:** Run a segmentation model every K frames; compute flow between frames with OFA; warp masks or intermediate features; use a lightweight CUDA/SM refinement and a confidence-triggered full re-segmentation path.

**Baselines and ablations:** Per-frame segmentation; key-frame propagation using CUDA optical flow; propagation without refinement; multiple K values; difficult sequences with occlusion, blur, articulated motion, and camera movement.

**Measure:** mIoU/IoU, boundary quality, temporal consistency, ID/part persistence if applicable, frames/s, p95 latency, Tensor/SM utilization, OFA utilization, VRAM, bandwidth, power/energy, and re-segmentation rate.

**Falsifiers:** No end-to-end improvement after conversion and synchronization; unacceptable boundary/occlusion failures; OFA unavailable or too low quality for the target resolution; or the CUDA baseline wins on both quality and whole-device cost.

## 2. OFA-assisted generative-video temporal consistency

**Evidence level:** Plausible extension informed by optical-flow-guided video research; not demonstrated here with NVIDIA OFA.

**Question:** Can OFA-derived motion cues improve temporal consistency in a generative-video pipeline without causing motion freezing, bias, or unacceptable overhead?

**Hypothesis:** A motion field or flow-derived confidence/conditioning signal can reduce temporal drift while the main model remains on Tensor/SM resources.

**Candidate design:** Generate a frame or short chunk; compute flow against a reference/generated predecessor; use warped features, motion masks, or a consistency controller in the next generation step. Start with an offline ablation before attempting asynchronous scheduling.

**Baselines and ablations:** No flow cue; CUDA optical flow; learned flow; flow at different resolutions; guidance strength; no-refinement and refinement paths; static and highly dynamic prompts.

**Measure:** FVD or task-appropriate video quality, framewise quality, temporal warping error, identity/subject consistency, motion diversity, human or automated artifact ratings, frames/s, latency, Tensor/SM/OFA utilization, VRAM, bandwidth, power, and regeneration rate.

**Falsifiers:** Quality improvement is not reproducible; the cue increases temporal artifacts or freezes motion; flow computation serializes generation; or conversion and synchronization dominate.

## 3. NVENC/NVDEC compressed visual memory

**Evidence level:** Plausible systems extension based on documented hardware video encode/decode and CUDA-resource paths.

**Question:** Can codec-backed storage of historical frames reduce memory pressure or SM work enough to improve whole-system behavior?

**Hypothesis:** Even if codec reconstruction is inferior to a learned latent history, dedicated encode/decode may make it useful in a memory-bound pipeline when raw frames would otherwise consume substantial VRAM or bandwidth.

**Candidate design:** Maintain tiers of recent raw frames and older encoded frames; decode references on demand; compare GPU-resident, host-resident, and storage-backed variants. Use application-allocated CUDA resources where supported.

**Baselines and ablations:** Raw frames; a standard image codec; learned/latent visual memory if available; different codecs, GOP/reference policies, resolutions, bit depths, and quality settings; encode/decode on versus off the critical path.

**Measure:** VRAM peak and average, compressed size, reconstruction quality, frame retrieval latency, frames/s, Tensor/SM work saved, NVENC/NVDEC utilization, memory/PCIe traffic, CPU use, power/energy, and end-to-end generation or analysis quality.

**Falsifiers:** Codec artifacts harm model quality; memory movement dominates; encode/decode stalls the model; or raw/latent baselines win on the complete metric set.

## 4. NVENC motion estimation versus OFA

**Evidence level:** Documented NVENC ME-only capability and documented NVIDIA guidance that OFA is preferable for computer vision/AI; comparative AI workload result is unknown.

**Question:** When block-level motion information is sufficient, can NVENC ME-only provide a useful or more available alternative to OFA?

**Hypothesis:** NVENC may be competitive for codec-oriented change detection, hybrid codecs, or platforms without OFA, even when OFA provides better visual matching.

**Candidate design:** Run supported NVENC motion estimation only and OFA on identical frame pairs and formats; compare vectors and downstream propagation/change-detection performance. Probe capability before running and record unsupported paths.

**Baselines and ablations:** CUDA optical flow; OFA presets/resolutions; NVENC partition and precision settings; complete encode versus ME-only; synchronous versus asynchronous API usage.

**Measure:** Vector accuracy or endpoint error where ground truth exists, downstream quality, latency, throughput, fixed-function utilization, Tensor/SM occupancy, memory traffic, power, and API/setup overhead.

**Falsifiers:** NVENC quality is insufficient for the target task; ME-only is unavailable under the target API/device; or OFA dominates once all overhead is included.

## 5. Fixed-function structural/anomaly side channels

**Evidence level:** Highly speculative systems extension; individual signals may be available from documented video or spatial APIs, but AI anomaly benefit is unproven.

**Question:** Can motion, residual, compressibility, or spatial-query signals detect coarse temporal failures early enough to improve a generative or analysis system?

**Hypothesis:** A cheap auxiliary structural signature may identify abrupt corruption, scene jumps, or discontinuity without duplicating the main model’s full computation.

**Candidate design:** Generate a signature per frame or chunk from flow statistics, motion-vector discontinuities, residual energy, reference-frame decisions, or a carefully scoped RT/OptiX spatial query. Train or calibrate thresholds only on a declared calibration split.

**Baselines and ablations:** No monitor; CUDA-only monitor; individual signal families; combined signature; threshold versus learned detector; regeneration disabled/enabled.

**Measure:** Detection precision/recall, false positive/negative rates, detection latency, recovery success, final quality, throughput, Tensor/SM and fixed-function utilization, VRAM, bandwidth, power, and added latency.

**Falsifiers:** Signals do not predict failures better than a cheap baseline; recovery costs more than it saves; or the monitor produces harmful false positives or semantic overclaims.

## 6. Experimental non-video tensor/KV compression

**Evidence level:** Highly speculative boundary experiment. Purpose-built KV compression remains the relevant benchmark, not an opponent to dismiss.

**Question:** Can a video-codec path for carefully laid-out activation or KV tensors yield a useful whole-system trade-off despite representation mismatch?

**Hypothesis:** A mediocre compressor on an otherwise-underused engine might reduce VRAM or SM work enough to help a memory-bound workload. The default expectation is that it will lose to purpose-built methods.

**Candidate design:** Map token position, attention head, and feature dimension onto several 2D/temporal layouts; encode/decode supported image/video formats; reconstruct to the original precision or a declared approximation; evaluate attention/output error.

**Baselines and ablations:** Uncompressed FP16/BF16 KV; standard KV quantization; a published purpose-built KV compression method; CUDA conversion only; codec path with and without host movement; multiple layouts and GOP/quality settings.

**Measure:** Compression ratio, VRAM, bandwidth, encode/decode latency, token throughput, p50/p95 latency, Tensor/SM/NVENC/NVDEC utilization, power/energy, perplexity, task accuracy, attention error, and long-context failure behavior.

**Falsifiers:** Conversion dominates; codec restrictions prevent a useful representation; reconstruction error harms quality; or purpose-built compression wins on quality, memory, and throughput. Such a result should be recorded as a useful boundary, not hidden.

## Suggested result format

Each completed experiment should add a short report containing:

1. hypothesis and preregistered decision rule;
2. exact environment and capability probes;
3. baseline/variant/ablations;
4. timeline and utilization evidence;
5. tables of quality, performance, memory, bandwidth, and power;
6. failure cases and negative results;
7. commands or scripts sufficient for another researcher to reproduce the run.
