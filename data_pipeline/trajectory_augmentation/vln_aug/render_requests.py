import json
from pathlib import Path

from PIL import Image


def validate_rendered_images(manifest_path: Path, episode_root: Path) -> dict:
    manifest = Path(manifest_path)
    root = Path(episode_root).resolve()
    missing = []
    invalid = []
    valid = []
    seen_request_ids = set()
    seen_paths = set()
    request_count = 0
    with manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            request = json.loads(line)
            request_count += 1
            request_id = request.get("request_id")
            relative_image_path = request.get("expected_image_relpath")
            if not request_id or request_id in seen_request_ids:
                invalid.append({"line": line_number, "reason": "missing or duplicate request_id"})
                continue
            if not relative_image_path or relative_image_path in seen_paths:
                invalid.append({"line": line_number, "reason": "missing or duplicate image path"})
                continue
            if not request.get("scene_id") or not request.get("camera_key"):
                invalid.append({"line": line_number, "reason": "scene_id and camera_key are required"})
                continue
            if len(request.get("body_pose_xyz_yaw", [])) != 4:
                invalid.append({"line": line_number, "reason": "body pose must have four values"})
                continue
            if not request.get("coordinate_metadata") or not request.get("camera_metadata"):
                invalid.append({"line": line_number, "reason": "coordinate and camera metadata are required"})
                continue
            seen_request_ids.add(request_id)
            seen_paths.add(relative_image_path)
            candidate = (root / request["expected_image_relpath"]).resolve()
            if root != candidate and root not in candidate.parents:
                invalid.append({"line": line_number, "reason": "image path escapes episode root"})
                continue
            if not candidate.is_file():
                missing.append({"request_id": request["request_id"], "path": str(candidate)})
                continue
            try:
                with Image.open(candidate) as image:
                    image.load()
                    width, height = image.size
                    channels = len(image.getbands())
                if (
                    height != int(request["expected_height"])
                    or width != int(request["expected_width"])
                    or channels != int(request["expected_channels"])
                ):
                    raise ValueError(
                        f"expected {request['expected_width']}x{request['expected_height']}x"
                        f"{request['expected_channels']}, got {width}x{height}x{channels}"
                    )
            except Exception as error:
                invalid.append(
                    {"request_id": request["request_id"], "path": str(candidate), "reason": str(error)}
                )
                continue
            valid.append({"request_id": request["request_id"], "path": str(candidate)})
    return {
        "complete": request_count > 0 and not missing and not invalid,
        "valid_count": len(valid),
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "valid": valid,
        "missing": missing,
        "invalid": invalid,
    }
