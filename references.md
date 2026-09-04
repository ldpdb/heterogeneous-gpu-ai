# References

References are grouped by what they establish. Vendor documentation establishes capabilities and API behavior; it does not establish the proposed end-to-end AI benefits. Research papers establish related algorithmic precedents; they do not establish that NVIDIA fixed-function hardware is the right implementation.

## NVIDIA documentation and primary technical sources

1. NVIDIA, [Optical Flow SDK](https://developer.nvidia.com/optical-flow-sdk). Overview of the SDK, hardware optical flow, documented use cases, and independence from CUDA cores on supported products.
2. NVIDIA, [NVOFA Tracker documentation](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-tracker/index.html). Describes a pipeline combining NVDEC, periodic GPU object detection, optical flow, and tracking.
3. NVIDIA, [Video Codec SDK: NVENC Video Encoder API Programming Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/index.html), especially “Motion Estimation Only Mode.” Documents ME-only capability queries, motion-vector output, and API constraints.
4. NVIDIA, [NVENC Application Note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-application-note/index.html). Summarizes NVENC hardware capabilities and motion-estimation use cases across listed GPU families.
5. NVIDIA, [Video Codec SDK Read Me and samples](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/read-me/index.html). Includes the AppEncME sample and SDK sample context.
6. NVIDIA, [Video Technologies documentation index](https://docs.nvidia.com/video-technologies/index.html). Entry point for current Video Codec and Optical Flow documentation and PyNvVideoCodec information.
7. NVIDIA, [CUDA C++ Programming Guide: Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html). Documents asynchronous data movement and overlap semantics.
8. NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), including concurrent copy and execution. Discusses copy-engine-dependent overlap and the `asyncEngineCount` device property.
9. NVIDIA, [Turing Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/). Describes RT-core roles, including BVH traversal and ray/triangle intersection, and their relationship to SM work.

## Related research

10. Baghbaderani et al., [Temporally-Consistent Video Semantic Segmentation With Bidirectional Occlusion-Guided Feature Propagation](https://openaccess.thecvf.com/content/WACV2024/html/Baghbaderani_Temporally-Consistent_Video_Semantic_Segmentation_With_Bidirectional_Occlusion-Guided_Feature_Propagation_WACV_2024_paper.html), WACV 2024. Relevant to flow-guided propagation, temporal consistency, occlusion, and low-cost segmentation trade-offs.
11. Jain et al., [Semantic Video Segmentation by Gated Recurrent Flow Propagation](https://arxiv.org/abs/1612.08871). Uses optical-flow-based temporal propagation with uncertainty-aware gating.
12. Liang et al., [FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_FlowVid_Taming_Imperfect_Optical_Flows_for_Consistent_Video-to-Video_Synthesis_CVPR_2024_paper.html), CVPR 2024. Relevant to using optical-flow clues in video synthesis while addressing imperfect flow.
13. Khaki et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750). A purpose-built KV-cache compression baseline and reminder that transformer-state compression has specialized methods and error trade-offs.
14. Hooper et al., [KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](https://arxiv.org/abs/2505.23416). Relevant to the non-video KV-cache compression comparison class.
15. NVIDIA, [OptiX](https://developer.nvidia.com/optix). Primary API/product entry point for experiments that may use RT-accelerated ray-tracing operations; consult versioned documentation for exact support.

## How to cite new results

When adding a benchmark result, cite the exact SDK/API documentation used, the GPU and driver, the software commit, the dataset and license, and any baseline paper or implementation. If a result is unpublished or locally measured, label it as an experiment result rather than as an established capability.
