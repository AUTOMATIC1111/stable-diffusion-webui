from .mediapipe_face_common import generate_annotation


def apply_mediapipe_face(image, max_faces: int = 1, min_confidence: float = 0.5):
    return generate_annotation(image, max_faces, min_confidence)
