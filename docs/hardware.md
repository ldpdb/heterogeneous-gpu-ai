# Hardware and API notes

This page separates vendor-documented behavior from interpretations for the research agenda. Exact support varies by GPU generation, platform, driver, API, codec, format, and SDK version. Always run a capability probe on the target system.

## Capability map

| Component | Documented capability | Evidence status for this project | Research relevance | Important caveat |
| --- | --- | --- | --- | --- |
| Optical Flow Accelerator | Computes optical-flow vectors between images on supported NVIDIA GPUs; NVIDIA describes hardware independence from CUDA cores and supplies SDK samples/tracker material. | Documented capability; tracker integration is a demonstrated use. | Motion, correspondence, mask propagation, temporal cues. | Flow quality and availability vary; motion is not semantics and fails around occlusion/deformation. |
| NVENC | Dedicated hardware video encoding; supported APIs also expose motion-estimation-only paths. | Documented capability. | Compressed history, codec metadata, motion statistics. | ME-only support is capability- and API-dependent; NVIDIA recommends OFA for AI/vision matching. |
| NVDEC | Dedicated hardware video decoding exposed through NVIDIA Video Codec APIs. | Documented capability. | Retrieval of compressed visual history. | Decode format, surface, synchronization, and memory-copy costs matter. |
| Copy/DMA engines | CUDA supports asynchronous data movement and overlap patterns; device properties expose engine-related capabilities. | Documented capability. | Staging, device/host transfers, pipeline overlap. | Separate engines do not eliminate memory/interconnect contention or synchronization. |
| RT cores | Accelerate selected ray-tracing operations such as BVH traversal and ray/triangle intersection through supported graphics/OptiX paths. | Documented capability. | Narrow spatial-query or structural side channel. | Not arbitrary tensor compute; acceleration structure, launch, and shading costs remain. |
| DLA/PVA/VIC and other SoC blocks | Product-specific specialized inference, vision, or image-processing functions on some NVIDIA platforms. | Documented at platform level; not a universal discrete-GPU assumption. | Platform-specific heterogeneous pipelines. | Treat as optional variants, not as properties of every NVIDIA GPU. |

## Optical Flow Accelerator (OFA)

NVIDIA’s Optical Flow SDK describes hardware-accelerated optical flow on Turing and later products and provides APIs, samples, and an object tracker. The NVOFA tracker example uses NVDEC for input, a GPU detector periodically, and optical flow between successive frames. The tracker then warps or matches regions using flow-derived information.

For this agenda, the demonstrated fact is the co-scheduling pattern: a semantic detector and a dedicated flow engine can appear in one pipeline. It does not demonstrate segmentation propagation or generative-video improvement. Those are proposed experiments.

Useful probe and reporting fields include GPU model, SDK version, supported input size/format, flow preset, grid size, forward/backward availability, global-flow support, and asynchronous completion behavior.

## NVENC

NVIDIA’s Video Codec SDK documents dedicated encode hardware and a motion-estimation-only mode on supported configurations. The API guide describes returning motion vectors and mode information, and describes possible uses such as motion-compensated filtering or custom/hybrid codecs. The same guide explicitly distinguishes this from OFA for computer vision, AI, and frame interpolation, where OFA vectors are recommended for better visual matching.

Do not describe ME-only mode as a general optical-flow engine. It produces codec-oriented motion information with its own block structure, precision, modes, and capability restrictions.

## NVDEC and memory paths

NVDEC is the decode side of NVIDIA’s Video Codec SDK. A compressed visual-memory experiment must distinguish:

- encoded bitstream storage location: VRAM, host RAM, or storage;
- decoded surface location and format;
- application-allocated CUDA resources versus copies through host memory;
- decode latency and synchronization with the model;
- reference-frame/GOP policy and error propagation;
- memory saved versus bandwidth and quality costs.

The right comparison is not just compression ratio. A codec path is useful only if the complete pipeline improves the target workload.

## Copy engines and overlap

CUDA documentation describes asynchronous data movement and concurrent copy/execute patterns. Measure actual overlap with a timeline or profiler rather than inferring it from asynchronous API calls. The number and behavior of copy engines are device-dependent, and copies still consume memory/interconnect resources.

For each experiment, report whether transfers are device-to-device, host-to-device, or device-to-host; whether host memory is pinned; which streams/events are used; and where the model waits.

## RT hardware

NVIDIA documents RT cores as accelerating BVH traversal and ray/triangle intersection testing. Application code remains responsible for ray generation and shading, and acceleration structures have construction/refit costs. A proposed RT experiment therefore needs a spatial representation and an API path—commonly OptiX or a graphics ray-tracing API—that can express the query.

The agenda does not claim that RT cores can accelerate arbitrary anomaly detection or tensor operations. A valid experiment would ask whether a narrow, naturally spatial query benefits after all setup and synchronization costs.

## Capability-probe checklist

Record at least:

- `nvidia-smi -q` or equivalent GPU/driver information;
- CUDA device properties, including compute capability, memory, and async-engine information;
- Video Codec SDK/NVENC/NVDEC capability queries;
- OFA SDK version, supported modes, formats, and resolutions;
- OptiX/graphics API and RT support if applicable;
- power limit, application clocks, persistence/power mode, and thermal state;
- exact benchmark commit and build flags.

The repository should accept “unsupported on this device” as a valid result. Do not silently substitute a CUDA implementation and then attribute its result to a fixed-function engine.
