# Fixed-function signals for multimodal video input

> **Epistemic status:** The hardware and API behaviors linked below are **documented capabilities**. NVIDIA's NVDEC–detector–OFA tracker is a **demonstrated use** of multiple engines in one vision pipeline. Every proposed effect on multimodal-model accuracy, latency, token use, or energy is a **plausible extension** or **speculative idea**, not a measured result of this repository.

## The question

Can a video-understanding system use specialized hardware to decide **which visual evidence to send to an expensive vision encoder**, or to attach useful temporal evidence, while preserving answer quality at lower end-to-end cost? The comparison must include an ordinary hardware-decoded pipeline: using NVDEC alone is already common and does not establish a new benefit.

```text
compressed input ── NVDEC ── decoded frames ── periodic coverage ──┐
                         │                                           │
                         ├─ codec statistics (when supported) ────────┤
                         └─ OFA frame-pair motion (optional) ────────┤
                                                                     ▼
                                         question-aware selector / controller
                                                                     │
                                       selected frames + timestamps + calibrated cues
                                                                     │
                                                 vision encoder → multimodal model
                                                                     │
                                    optional focused revisit of a source-video interval
```

The selector or decision model runs on a CPU, CUDA/SM, Tensor-capable path, or another supported inference engine. The fixed-function blocks supply signals; they do not recognize objects, decide what matters to a question, or verify a model's answer. NVIDIA's [tracker example](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-tracker/index.html) already combines NVDEC, a GPU object detector, and OFA. The proposed multimodal frame-selection and feedback effects remain untested here. Related research already explores [question-aware frame selection](https://arxiv.org/abs/2607.01737) and [flow-informed video understanding](https://arxiv.org/abs/2510.05836), so novelty should be claimed only for a measured systems result, if one emerges.

## Assessment by engine

| Engine or block | Documented capability | Candidate contribution and evidence level | Boundary or decision |
| --- | --- | --- | --- |
| **NVDEC** | Decode supported bitstreams into video surfaces; the Video Codec SDK 13.1 guide also documents capability-gated H.264/HEVC per-block decode statistics, including motion vectors, quantization parameters, and coding-unit types. | **Plausible extension:** use existing compressed-stream statistics to propose intervals for closer inspection. Compare with ordinary NVDEC output, uniform frames, and simple image differences. | These are codec decisions, not semantic events; codec, driver, surface, and API support must be probed. A pipeline already using NVDEC gains nothing merely by naming it. [Decoder guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvdec-video-decoder-api-prog-guide/index.html) |
| **OFA** | Compute optical-flow vectors between supported frame pairs; the API can expose flow cost and, on supported paths, global flow. | **Plausible extension:** estimate motion concentration, track known regions, prioritize temporal intervals, or supply motion features to a trained video model. | It requires suitable decoded frame pairs and a separate interpretation step. Camera motion, occlusion, small objects, cuts, and low-motion semantic changes can defeat a motion-only selector. [OFA guide](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-programming-guide/index.html) |
| **NVENC** | Encode supported video; capability-gated motion-estimation-only mode returns codec-oriented motion vectors. | **Plausible extension:** create a lower-cost proxy/history stream when repeated video access justifies encoding, or compare ME-only vectors with OFA. | Re-encoding an already compressed source solely to choose frames may add more work than it saves. NVENC is optional in the input pipeline; NVIDIA recommends OFA for vision/AI matching. [Encoder guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-video-encoder-api-prog-guide/index.html) |
| **Copy engines** | Support asynchronous transfers and possible copy/compute overlap on capable devices. | **Plausible systems aid:** stage selected frames or results while other work runs. | They move data; they do not extract meaning. Measure traffic and a timeline before claiming overlap. [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) |
| **RT hardware** | Accelerate supported ray-tracing traversal/intersection paths. | **Speculative, restricted:** a video system already maintaining 3D scene geometry could test a spatial query. | No natural role is established for choosing frames in ordinary 2D video. Building a scene or acceleration structure merely to use RT hardware is not a recommended starting experiment. [NVIDIA architecture description](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/) |
| **Jetson VIC / PVA / DLA** | On supported Jetson/DRIVE products, VIC handles selected image operations, PVA supports selected vision processing, and DLA runs compatible inference subgraphs under SDK restrictions. | **Platform-specific plausible extension:** offload resize/color conversion, a supported vision operation, or a small selector model while the GPU runs the main model. | These blocks are not properties of desktop RTX GPUs. Verify exact device, SDK, algorithm/layer support, fallback behavior, copies, and power. [VPI backends](https://docs.nvidia.com/vpi/basic_concepts.html), [PVA SDK](https://docs.nvidia.com/pva/index.html), [TensorRT DLA restrictions](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) |

## Three testable directions

1. **Motion- and question-aware frame selection — plausible extension.** Preserve sparse periodic coverage, then allocate a fixed additional frame budget using codec statistics, OFA summaries, and a question-aware controller. Compare equal-budget answer quality and event recall with uniform, scene-change, simple frame-difference, and semantic-selection baselines. Motion is a candidate signal, not a proxy for importance.
2. **Temporal features accompanying selected frames — speculative idea.** Encode OFA-derived flow or tracked-region summaries with timestamps and visual features. Train or adapt a model with this representation; raw flow vectors are not meaningful language tokens by themselves. Compare with the same model and frame budget without flow, and with a programmable or learned-flow baseline. Assess camera movement, occlusion, and inference overhead.
3. **Focused revisit — plausible systems extension.** After an initial answer or uncertainty signal, decode a short source interval at finer temporal resolution. Compare against a single-pass system with the same total frame and compute budget. Record seek/GOP costs, extra model calls, decision errors, and whether the revisit actually changes correct answers. This can use NVDEC without OFA if cheaper signals suffice.

An auxiliary decision layer can route `accept / inspect more / abstain` using calibrated uncertainty, but it is an ordinary model or rule-based controller, not a capability of OFA, NVENC, or NVDEC. It should be evaluated for false confidence and unnecessary revisits. No claim is made here about a specific external decision-model architecture.

## Failure cases to include

Use public, licensed video and question sets that contain camera pans, cuts, static scenes with changing text, small or slow actions, occlusion, and events outside high-motion intervals. Report event localization as well as answer accuracy: a selector can appear efficient while silently omitting the only relevant moment. Account for decode, surface mapping, format conversion, OFA setup/flow, statistics extraction, copies, selector inference, model inference, and any second pass. See [experiment cards 7–9](../EXPERIMENTS.md) for baselines and rejection criteria.
