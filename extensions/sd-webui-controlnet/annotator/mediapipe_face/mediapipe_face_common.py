from typing import Mapping

import mediapipe as mp
import numpy


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection  # Only for counting faces.
mp_face_mesh = mp.solutions.face_mesh
mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS
mp_body_connections = mp.solutions.pose_connections.POSE_CONNECTIONS

DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
PoseLandmark = mp.solutions.drawing_styles.PoseLandmark

min_face_size_pixels: int = 64
f_thick = 2
f_rad = 1
right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

# mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec = {}
for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec[edge] = head_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
    face_connection_spec[edge] = left_eye_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
    face_connection_spec[edge] = left_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
#    face_connection_spec[edge] = left_iris_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
    face_connection_spec[edge] = right_eye_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
    face_connection_spec[edge] = right_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
#    face_connection_spec[edge] = right_iris_draw
for edge in mp_face_mesh.FACEMESH_LIPS:
    face_connection_spec[edge] = mouth_draw
iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}


def draw_pupils(image, landmark_list, drawing_spec, halfwidth: int = 2):
    """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
    landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be H,W,C.")
    image_rows, image_cols, image_channels = image.shape
    if image_channels != 3:  # BGR channels
        raise ValueError('Input image must contain three channel bgr data.')
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
                (landmark.HasField('visibility') and landmark.visibility < 0.9) or
                (landmark.HasField('presence') and landmark.presence < 0.5)
        ):
            continue
        if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
            continue
        image_x = int(image_cols*landmark.x)
        image_y = int(image_rows*landmark.y)
        draw_color = None
        if isinstance(drawing_spec, Mapping):
            if drawing_spec.get(idx) is None:
                continue
            else:
                draw_color = drawing_spec[idx].color
        elif isinstance(drawing_spec, DrawingSpec):
            draw_color = drawing_spec.color
        image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
    # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
    return image[:, :, ::-1]


def generate_annotation(
        img_rgb,
        max_faces: int,
        min_confidence: float
):
    """
    Find up to 'max_faces' inside the provided input image.
    If min_face_size_pixels is provided and nonzero it will be used to filter faces that occupy less than this many
    pixels in the image.
    """
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
    ) as facemesh:
        img_height, img_width, img_channels = img_rgb.shape
        assert(img_channels == 3)

        results = facemesh.process(img_rgb).multi_face_landmarks

        if results is None:
            print("No faces detected in controlnet image for Mediapipe face annotator.")
            return numpy.zeros_like(img_rgb)

        # Filter faces that are too small
        filtered_landmarks = []
        for lm in results:
            landmarks = lm.landmark
            face_rect = [
                landmarks[0].x,
                landmarks[0].y,
                landmarks[0].x,
                landmarks[0].y,
            ]  # Left, up, right, down.
            for i in range(len(landmarks)):
                face_rect[0] = min(face_rect[0], landmarks[i].x)
                face_rect[1] = min(face_rect[1], landmarks[i].y)
                face_rect[2] = max(face_rect[2], landmarks[i].x)
                face_rect[3] = max(face_rect[3], landmarks[i].y)
            if min_face_size_pixels > 0:
                face_width = abs(face_rect[2] - face_rect[0])
                face_height = abs(face_rect[3] - face_rect[1])
                face_width_pixels = face_width * img_width
                face_height_pixels = face_height * img_height
                face_size = min(face_width_pixels, face_height_pixels)
                if face_size >= min_face_size_pixels:
                    filtered_landmarks.append(lm)
            else:
                filtered_landmarks.append(lm)

        # Annotations are drawn in BGR for some reason, but we don't need to flip a zero-filled image at the start.
        empty = numpy.zeros_like(img_rgb)

        # Draw detected faces:
        for face_landmarks in filtered_landmarks:
            mp_drawing.draw_landmarks(
                empty,
                face_landmarks,
                connections=face_connection_spec.keys(),
                landmark_drawing_spec=None,
                connection_drawing_spec=face_connection_spec
            )
            draw_pupils(empty, face_landmarks, iris_landmark_spec, 2)

        # Flip BGR back to RGB.
        empty = reverse_channels(empty).copy()

        return empty
