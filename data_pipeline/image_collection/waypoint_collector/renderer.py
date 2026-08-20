import time

import numpy as np

from waypoint_collector.airsim_session import AirSimSessionUnavailableError


class EpisodeRenderError(RuntimeError):
    pass


class SessionRenderError(EpisodeRenderError):
    pass


class InvalidFrameRenderError(EpisodeRenderError):
    """An AirSim response whose image payload is unusable for rendering."""
    def __init__(self, message, frames=None, invalid_view=None):
        super().__init__(message)
        self.frames = {} if frames is None else {
            str(view): frame.copy() for view, frame in frames.items()
        }
        self.invalid_view = None if invalid_view is None else str(invalid_view)

def validate_rgb_frames(frames, views, image_width=224, image_height=224):
    if set(frames) != set(views):
        raise EpisodeRenderError(
            "expected RGB views {}, received {}".format(
                tuple(views), tuple(sorted(frames))
            )
        )
    for view in views:
        frame = frames[view]
        if not isinstance(frame, np.ndarray):
            raise EpisodeRenderError("{} frame is not a numpy array".format(view))
        expected_shape = (int(image_height), int(image_width), 3)
        if frame.shape != expected_shape:
            raise EpisodeRenderError(
                "{} frame has shape {}, expected {}x{}x3".format(
                    view, frame.shape, image_width, image_height,
                )
            )
        if frame.dtype != np.uint8:
            raise EpisodeRenderError("{} frame must use uint8".format(view))
        if frame.size == 0:
            raise EpisodeRenderError("{} frame is empty".format(view))
    return True


def render_episode(session, requests, camera_records, views, sink,
                   frame_attempts=10, retry_delay_seconds=0.0,
                   position_tolerance=0.01, rotation_tolerance_degrees=0.1,
                   image_width=224, image_height=224):
    if int(frame_attempts) < 1:
        raise ValueError("frame_attempts must be positive")
    try:
        session.apply_camera_records(camera_records)
    except AirSimSessionUnavailableError as error:
        raise SessionRenderError(
            "camera setup failed because the AirSim session is unavailable: {}".format(
                error
            )
        ) from error
    rendered = 0
    for request in requests:
        last_error = None
        for attempt in range(int(frame_attempts)):
            frames = None
            try:
                session.set_vehicle_pose(request)
                position_error, rotation_error = session.verify_vehicle_pose(request)
                if (
                    not np.isfinite(position_error)
                    or not np.isfinite(rotation_error)
                ):
                    raise EpisodeRenderError(
                        "non-finite vehicle pose readback error: "
                        "position={}, rotation={}".format(
                            position_error, rotation_error
                        )
                    )
                if position_error > float(position_tolerance):
                    raise EpisodeRenderError(
                        "vehicle position readback error {:.6f} exceeds {:.6f}".format(
                            position_error, position_tolerance
                        )
                    )
                if rotation_error > float(rotation_tolerance_degrees):
                    raise EpisodeRenderError(
                        "vehicle rotation readback error {:.6f} exceeds {:.6f}".format(
                            rotation_error, rotation_tolerance_degrees
                        )
                    )
                frames = session.capture_rgb(views)
                validate_rgb_frames(
                    frames, views, image_width=image_width,
                    image_height=image_height,
                )
                last_error = None
                break
            except Exception as error:
                if isinstance(error, InvalidFrameRenderError):
                    error.frames = {
                        str(view): frame.copy()
                        for view, frame in (frames or {}).items()
                    }
                last_error = error
                if attempt + 1 < int(frame_attempts) and retry_delay_seconds:
                    time.sleep(float(retry_delay_seconds))
        if last_error is not None:
            if isinstance(last_error, AirSimSessionUnavailableError):
                error_type = SessionRenderError
            elif isinstance(last_error, InvalidFrameRenderError):
                error_type = InvalidFrameRenderError
            else:
                error_type = EpisodeRenderError
            error = error_type(
                "request {} failed after {} attempts: {}".format(
                    request.request_id, frame_attempts, last_error
                )
            )
            if isinstance(last_error, InvalidFrameRenderError):
                error.frames = last_error.frames
                error.invalid_view = last_error.invalid_view
            raise error from last_error
        sink.append(request, frames)
        rendered += 1
    return rendered
