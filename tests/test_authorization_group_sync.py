from types import SimpleNamespace

from core.selective_camera import MainCamera


def _camera():
    camera = object.__new__(MainCamera)
    camera.config = SimpleNamespace(
        FACE_AUTH_CONFIRM_MATCHES=1,
        FACE_AUTH_REVOKE_MISMATCHES=5,
        FACE_AUTH_HOLD_SECONDS=0.8,
        MATCH_THRESHOLD=0.45,
        FACE_AUTH_RETAIN_THRESHOLD=0.30,
        FACE_AUTH_TRACK_IOU=0.12,
    )
    return camera


def _face(track_id, score, bbox=(100, 80, 220, 220)):
    return {
        'track_id': track_id,
        'bbox': list(bbox),
        'similarity': score,
        'gallery_valid': True,
        'group': 'master',
    }


def _observation(identity):
    return {'identity': identity}


def test_strong_registered_match_is_stored_as_internal():
    camera = _camera()
    faces = [_face(1, 0.62)]
    states = camera._update_display_authorization(
        faces, [_observation('employee-a')], {}, 10.0
    )

    assert faces[0]['authorized'] is True
    assert faces[0]['group'] == 'internal'
    assert states


def test_profile_score_drop_keeps_authorized_internal_across_track_reset():
    camera = _camera()
    first = [_face(1, 0.62)]
    states = camera._update_display_authorization(
        first, [_observation('employee-a')], {}, 10.0
    )

    # Simulate a head turn: ArcFace score drops but remains above retain threshold,
    # and the geometric tracker assigns a new track id.
    second = [_face(99, 0.33, bbox=(104, 82, 224, 222))]
    states = camera._update_display_authorization(
        second, [_observation('employee-a')], states, 10.1
    )

    assert second[0]['authorized'] is True
    assert second[0]['group'] == 'internal'
    assert states


def test_unrecognized_detected_face_is_always_external_not_master():
    camera = _camera()
    faces = [_face(7, None)]
    camera._update_display_authorization(
        faces, [_observation(None)], {}, 20.0
    )

    assert faces[0]['authorized'] is False
    assert faces[0]['group'] == 'external'
