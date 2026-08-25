from __future__ import annotations

from pathlib import Path
from threading import local
from typing import Any, Iterable

import numpy as np
from rich.progress import Progress

from tool.navvla.cli.generate_visual_cache import (
    LazyEncoderUnavailable,
    build_ref_metadata_lookup,
    extract_qwen3_visual_tokens_batch,
    find_missing_cache_refs,
    iter_prefetched_batches,
    iter_video_ordered_image_batches,
    load_existing_ref_rows_for_refs,
    load_visual_cache_refs,
    order_refs_by_video,
    rebuild_mmap_profile_index,
    resolve_ref_metadata,
    unpack_visual_token_output,
    write_mmap_checkpoint_index_rows,
    _load_dataset_lookup_tables,
)
from tool.navvla.visual_token_cache import (
    MMAP_NPY_VISUAL_TOKEN_FORMAT,
    MMapNpyProfileShardWriter,
    VisualTokenProfile,
    write_profile_index,
    write_profile_manifest,
    write_profile_token_record,
)
from tool.navvla.workers import resolve_workers


class VisualTokenEncoderUnavailable(RuntimeError):
    pass


def generate_profile_cache_parallel(
    dataset_root: Path,
    *,
    profile: VisualTokenProfile,
    refs: Iterable[str] | None = None,
    encoder: Any | None = None,
    encoder_factory: Any | None = None,
    workers: int | None = None,
    skip_existing: bool = False,
    batch_size: int = 1,
    prefetch_batches: int = 2,
    input_resize: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if encoder is None and encoder_factory is None:
        raise VisualTokenEncoderUnavailable(
            "visual token cache generation requires visual_token_encoder or visual_token_encoder_factory"
        )

    dataset_root = Path(dataset_root)
    all_refs = list(load_visual_cache_refs(dataset_root) if refs is None else refs)
    existing_rows = load_existing_ref_rows_for_refs(dataset_root, profile.name, all_refs) if skip_existing else []
    existing_refs = {str(row["ref"]) for row in existing_rows}
    missing_refs = find_missing_cache_refs(all_refs, existing_refs=existing_refs)

    write_profile_manifest(dataset_root, profile)
    data, episodes, video_index, cameras, info = _load_dataset_lookup_tables(dataset_root)
    lookup = build_ref_metadata_lookup(data=data, episodes=episodes, video_index=video_index)
    print(
        f"prepared lookup tables: refs={len(missing_refs)} episodes={len(lookup['episode_by_id'])} "
        f"frames={len(lookup['data_by_episode_frame'])} video_rows={len(lookup['video_by_data_key'])}",
        flush=True,
    )
    metadata_by_ref = {ref: resolve_ref_metadata(ref, cameras=cameras, lookup=lookup) for ref in missing_refs}
    print(f"resolved metadata for {len(metadata_by_ref)} refs; ordering by video", flush=True)
    ordered_refs = order_refs_by_video(missing_refs, metadata_by_ref=metadata_by_ref)
    worker_state = local()

    def get_encoder() -> Any:
        if encoder is not None:
            return encoder
        cached = getattr(worker_state, "encoder", None)
        if cached is None:
            try:
                cached = encoder_factory()
            except LazyEncoderUnavailable as exc:
                raise VisualTokenEncoderUnavailable(str(exc)) from exc
            worker_state.encoder = cached
        return cached

    generated_rows: list[dict[str, object]] = []

    def flush_mmap_checkpoint_rows(rows: list[dict[str, object]]) -> None:
        write_mmap_checkpoint_index_rows(dataset_root, profile.name, rows)
        rebuild_mmap_profile_index(dataset_root, profile)

    mmap_writer = (
        MMapNpyProfileShardWriter(dataset_root, profile=profile, on_flush=flush_mmap_checkpoint_rows)
        if profile.file_format == MMAP_NPY_VISUAL_TOKEN_FORMAT
        else None
    )
    with Progress() as progress:
        task = progress.add_task(f"visual-cache {profile.name}", total=len(ordered_refs))
        for batch_refs, batch_metas, batch_images in iter_prefetched_batches(
            iter_video_ordered_image_batches(
                dataset_root,
                refs=ordered_refs,
                metadata_by_ref=metadata_by_ref,
                info=info,
                batch_size=batch_size,
                input_resize=input_resize,
            ),
            max_prefetch_batches=prefetch_batches,
        ):
            batch_outputs = extract_qwen3_visual_tokens_batch(batch_images, encoder=get_encoder())
            if len(batch_outputs) != len(batch_refs):
                raise ValueError(f"encoder batch output length {len(batch_outputs)} does not match input batch {len(batch_refs)}")
            for ref, meta, output in zip(batch_refs, batch_metas, batch_outputs):
                image_embeds, deepstack_embeds, output_metadata = unpack_visual_token_output(output)
                row_metadata = {
                    "episode_id": meta["episode_id"],
                    "trajectory_id": meta["trajectory_id"],
                    "frame_index": meta["frame_index"],
                    "source_frame_index": meta["source_frame_index"],
                    "data_index": meta["data_index"],
                    "camera_name": meta["camera_name"],
                    "video_key": meta["video_key"],
                    "dataset_name": meta["dataset_name"],
                    "split": meta["split"],
                    **output_metadata,
                }
                if mmap_writer is not None:
                    if deepstack_embeds is not None:
                        raise ValueError(f"profile {profile.name} mmap_npy cache does not support deepstack_embeds")
                    mmap_writer.add(ref=ref, image_embeds=np.asarray(image_embeds), metadata=row_metadata)
                else:
                    record = write_profile_token_record(
                        dataset_root,
                        profile=profile,
                        ref=ref,
                        image_embeds=np.asarray(image_embeds),
                        deepstack_embeds=None if deepstack_embeds is None else np.asarray(deepstack_embeds),
                    )
                    generated_rows.append({"ref": record.ref, "path": record.path, **row_metadata})
            progress.advance(task, advance=len(batch_refs))
    if mmap_writer is not None:
        generated_rows.extend(mmap_writer.close())

    rows = list(existing_rows) + generated_rows
    index_path = write_profile_index(dataset_root, profile.name, rows)
    return {
        "generated_by_writer": True,
        "profile": profile.name,
        "index": str(index_path),
        "records": len(rows),
        "generated_records": len(generated_rows),
        "skipped_existing_records": len(existing_rows),
        "workers": resolve_workers(workers),
        "batch_size": int(batch_size),
        "prefetch_batches": int(prefetch_batches),
        "input_resize": None if input_resize is None else f"{input_resize[0]}x{input_resize[1]}",
    }
