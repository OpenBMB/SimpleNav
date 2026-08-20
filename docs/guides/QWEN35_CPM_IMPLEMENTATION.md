# NavVLA Qwen3.5 CPM

`navvla_qwen35_cpm` keeps the `navvla_cpm` action head, TVI, BATS, long-memory,
state/action normalization, and train/eval contracts. It replaces only the VLM
backbone path with Qwen3.5-VL.

## Visual Path

The supported cache contract is:

```text
256x256 image
  -> Qwen3.5 patch embedding and ViT blocks
  -> 256 x vision_hidden pre-merge tokens
  -> frozen Qwen3.5 checkpoint merger
  -> 64 x llm_hidden post-merge tokens (current frame, no NavVLA pool)
  -> FP32 2D adaptive pooling on the history-cache branch only
  -> 4 x llm_hidden BF16 history tokens
  -> uint16 BF16 bit view in mmap .npy cache
  -> Qwen3.5 input embedding scatter
  -> Qwen3.5 M-RoPE and language_model.forward
```

The action-query suffix uses Qwen3.5's declared single-token FIM markers
(`<|fim_prefix|>`, repeated `<|fim_pad|>`, `<|fim_suffix|>`). Model startup
validates that a local checkpoint's tokenizer preserves this contract.

The 2D pool remains after the merger. The entire visual tower, including the
merger, is frozen and held in eval mode. Offline generation and online history updates therefore cache
the final four pooled history tokens. Cached history is decoded from BF16 bits and consumed
directly without running the ViT, merger, or a second pool. A current online image
uses the raw 64-token merger output directly; the same output is pooled separately
to four tokens when retained as history.

For the official 4B config, the ViT hidden size is 1024 and the LLM hidden size
is 2560. The final cache is 4 x 2560 x 2 bytes = 20 KiB per image,
or 80 KiB for four cameras per frame. Across the currently indexed AerialVLN,
OpenFly, and TravelUAV camera references, the raw tensor estimate is about
122.4 GiB before filesystem/index overhead.

## Cache Generation

Generate a cache with the same resize and checkpoint used by training:

```bash
python -m tool.navvla.cli.generate_visual_cache DATASET_ROOT \
  --profile qwen3_5_4b_postmerge_pool4_256_mmap \
  --visual-head qwen3_5_postmerge_pool4 \
  --encoder-name Qwen3.5-4B \
  --encoder-ckpt local/models/Qwen3.5-4B \
  --token-level vit_postmerge_pool4 \
  --token-count 4 \
  --hidden-dim 0 \
  --dtype uint16 \
  --shard-size 8192 \
  --input-resize 256x256 \
  --camera-names front \
  --file-format mmap_npy
```

The manifest records the checkpoint, cache stage, resize, patch size, merger
size, `dtype=uint16`, `storage_encoding=bfloat16_bits`, and inferred LLM hidden size. Each index row also records
`grid_t`, `grid_h`, `grid_w`, and `cache_stage`. The loader rejects missing or
mixed stages, verifies that each cached row has four tokens, and passes the manifest's
`encoder_ckpt` to the model for an exact checkpoint-contract check.

Online inference emits the same structured payload (`tokens`, `grid_thw`,
`cache_stage`, `visual_token_profile`, `encoder_ckpt`, and `storage_encoding`) and resizes images to
256x256 with bicubic interpolation before the Qwen3.5 processor. Runtime history and
long-memory updates preserve this payload. Offline and online generation apply
the same FP32 pooling, BF16 conversion, and uint16 bit-view order; cached entries then
enter the language-model path directly.

Training, inference, and offline cache generation all request FlashAttention 2.
`vision_id` remains disabled, so no numbered picture text is inserted; Qwen's
fixed vision boundary tokens remain in the normal multimodal sequence.

Qwen3.5 visual forward is intentionally executed one image at a time in both
paths. Batched Qwen visual execution can differ from single-image execution, so
isolating each image is required for bitwise-identical online
and offline cache values. `BATCH_SIZE` still controls video decode, preprocessing,
and cache-write batching, but it does not combine images in one ViT forward.

New BF16-bit caches must be generated under the 256 profile. Legacy FP16 pool4
manifests remain readable through the numeric-conversion compatibility branch;
they are not rewritten in place. The training config and cache generator must
use the same checkpoint path string because runtime checks this field exactly.

For the repository's AerialVLN, OpenFly, and TravelUAV training roots, use
`tool/navvla/generate_qwen35_postmerge_caches.sh`. It preserves every existing
cache profile, resumes only this post-merge pool4 profile, filters TravelUAV to
the four training cameras, and refuses a run when its estimated cache plus the
configured free-space reserve does not fit the available filesystem.

```bash
# Read-only capacity and manifest check.
PREFLIGHT_ONLY=1 tool/navvla/generate_qwen35_postmerge_caches.sh

# Resume one dataset on four visible GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 DATASETS=AerialVLN NUM_GPUS=4 \
  tool/navvla/generate_qwen35_postmerge_caches.sh

# Small real-data smoke run; LIMIT applies per selected dataset.
DATASETS=OpenFly,TravelUAV LIMIT=1 BATCH_SIZE=1 \
  tool/navvla/generate_qwen35_postmerge_caches.sh


  bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml
```

Use
`examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml` as the training
template. Replace the checkpoint and dataset paths with local paths. The
configured freeze list freezes the complete `visual` module, including merger.
