# SimpleNAV Vision and Roadmap

[Back to the main README](../../README.md) · [中文](VISION_AND_ROADMAP_ZH.md)

## Vision

SimpleNAV aims to become a general-purpose research framework for Vision-and-Language Navigation. It is not tied to one dataset, robot platform, visual-language backbone, action space, or simulator. Stable intermediate protocols and extensible components connect the full research lifecycle:

```text
data acquisition and conversion
  -> unified navigation representation
  -> composable model development
  -> single-domain and multi-domain training
  -> open-loop and closed-loop evaluation
  -> results, failures, and deployment feedback
  -> the next data and model iteration
```

“General-purpose” does not require every task to share identical inputs, actions, or metrics. It requires each task to preserve its semantics through an explicit adapter while reusing shared infrastructure.

## Five dimensions of extensibility

### Data

- Adapt aerial, indoor, ground, driving, and real-platform navigation data.
- Support single-view, multi-view, video, state, map, and other navigation conditions.
- Record coordinate, time, action, and camera semantics as verifiable protocols.
- Support conversion, trajectory augmentation, simulator rerendering, statistics, and quality reports.
- Publish preparation manifests for restricted datasets instead of redistributing protected content.

### Models

- Replace the visual-language backbone without rewriting data and training pipelines.
- Compose history selection, short- and long-term memory, spatiotemporal encoding, and world models.
- Support continuous waypoints, discrete actions, and future hybrid action interfaces.
- Keep model capabilities separate from benchmark-specific exceptions.

### Training

- Support single-dataset, mixture, and cross-domain joint training.
- Manage normalization, sampling ratios, action contracts, and checkpoint metadata consistently.
- Support single-node, multi-GPU, multi-node, resume, and reproducible resolved configs.
- Preserve clear interfaces for pretraining, supervised fine-tuning, preference optimization, and online adaptation.

### Evaluation

- Isolate input formats, task semantics, and metrics through benchmark adapters.
- Isolate AirSim, Habitat, Unreal, and future real-platform connections through simulator backends.
- Support offline action checks, teacher forcing, open-loop evaluation, and closed-loop rollout.
- Record resolved config, run plan, episode artifacts, failures, and aggregate metrics uniformly.
- Add a benchmark without modifying the common rollout loop.

### Ecosystem

- Release data tools, model cards, training recipes, evaluation templates, and reproduction reports.
- Maintain stable entry points for data, model, benchmark, and deployment modules.
- Define contribution boundaries, tests, and third-party provenance.
- Grow a comparable and maintainable open VLN research ecosystem.

## Design principles

### Share infrastructure, preserve task semantics

Storage, loading, training, configuration, and artifact organization are shared. Coordinates, actions, and metrics that cannot be unified or inferred must be converted explicitly by adapters.

### Extend through interfaces

A new dataset, model, or benchmark should implement an adapter or component. Dataset names and path conditions should not accumulate inside common loops.

### Reproduction is a first-class artifact

Each result should link at least the source revision and environment, dataset version and manifest, resolved training config, checkpoint and model card, evaluation config and simulator asset summary, and raw episode and aggregate results.

### Separate current capability from future direction

Roadmap items are not claims of present support. Releases must distinguish implemented, validated, preparing-for-release, and long-term research work.

## Milestones

### Release 01 — reproducible foundations

- Unify LeRobot v3 navigation data and action contracts.
- Provide conversion, validation, repair, statistics, history-index, and visual-cache tools.
- Provide the Qwen3.5-VL navigation model path.
- Provide single-dataset and initial multi-dataset recipes.
- Establish AirSim, Habitat, and offline evaluation infrastructure.
- Maintain public checkpoints, model cards, configs, and benchmark results.

### Release 02 — modular VLN development framework

- Maintain the integrated [`data_pipeline/`](../../data_pipeline/README.md) toolkit for conversion, augmentation, and simulator collection.
- Stabilize dataset-adapter, model-component, and benchmark-plugin interfaces.
- Add more VLM backbones, memory modules, and action heads.
- Standardize model cards, dataset cards, and reproduction manifests.
- Provide portable simulator setup, asset validation, and smoke tests.
- Improve cross-dataset mixture training and per-domain monitoring.

### Release 03 — cross-domain general navigation

- Expand aerial, indoor, ground, driving, and real-platform data.
- Research cross-platform state and action representations.
- Support multi-domain joint training and domain- or task-conditioned models.
- Evaluate unseen scenes, instructions, platforms, and cross-simulator transfer systematically.
- Explore world models, hierarchical planning, and online memory.

### Long term — general embodied navigation and real deployment

- Build a navigation VLA that handles diverse instructions, environments, and platforms.
- Connect navigation and manipulation data while preserving task action semantics.
- Unify simulator, offline-data, and real-platform evaluation and deployment interfaces.
- Improve real-time control, safety constraints, failure recovery, and continual learning.
- Grow a community-extensible ecosystem of data, models, and benchmarks.

## Verifiable milestone artifacts

| Addition | Minimum evidence |
| --- | --- |
| Dataset | Adapter, semantic contract, validation report, minimal sample |
| Model | Config, architecture description, checkpoint contract, smoke test |
| Training recipe | Resolved config, data statistics, logs, checkpoint |
| Benchmark | Input adapter, run config, episode artifact, metric tests |
| Simulator backend | Environment check, version record, minimal closed loop, failure handling |
| Public result | Checkpoint, source revision, configs, data version, raw summary |
