"""Sample code for processing a face image
"""
import yk_face as YKF


KEY = 'subscription key'  # Replace with a valid Subscription Key here.
YKF.Key.set(KEY)

BASE_URL = 'YouFace API URL'  # Replace with a valid API URL here.
YKF.BaseUrl.set(BASE_URL)

# Detect faces in an image
img_file_path = 'detection1.jpg'
detected_faces = YKF.face.process(img_file_path)
print(f'Detected faces: {detected_faces}')

# Compute matching score of two processed faces, given their biometric templates
matching_score = YKF.face.verify(detected_faces[0]['template'], detected_faces[0]['template'])
print(f'Verify - Matching score: {matching_score}')

matching_score = YKF.face.verify_images(img_file_path, img_file_path)
print(f"Verify images - Matching score: {matching_score}")

# Create a group
group_id = 'demo_group'
YKF.group.create(group_id)

# Add a person to a group
person_id = 'demo_person'
YKF.group.add_person(group_id=group_id, person_id=person_id, face_template=detected_faces[0]['template'])

# Get a previously added person biometric template
template = YKF.group.get_person_template(group_id=group_id, person_id=person_id)
print(f'{person_id} biometric template: {template}')

# List all person ids in a group
ids_in_group = YKF.group.list_ids(group_id)
print(f'Person ids in {group_id}: {ids_in_group}')

# Check if a face template belongs to a specific person
verify_id_score = YKF.face.verify_id(
    face_template=detected_faces[0]['template'],
    person_id=person_id,
    group_id=group_id
)
print(f'Verify ID - Matching score: {verify_id_score}')

# Identify an unknown person (using her biometric template) in a group
identification_candidates = YKF.face.identify(face_template=detected_faces[0]['template'], group_id=group_id)
print(f'Identification candidates: {identification_candidates}')

# Remove a person from a group
YKF.group.remove_person(group_id=group_id, person_id=person_id)
new_identification_candidates = YKF.face.identify(face_template=detected_faces[0]['template'], group_id=group_id)
print(f'Identification candidates (empty group): {new_identification_candidates}')

# Delete a group
YKF.group.delete(group_id)
