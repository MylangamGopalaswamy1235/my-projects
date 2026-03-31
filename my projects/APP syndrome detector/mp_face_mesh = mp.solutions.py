mp_face_mesh = mp.solutions.face_mesh

def analyze_face_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return [{'syndrome': 'Error', 'confidence': 0, 'note': 'Image not found or invalid.'}]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return [{'syndrome': 'Healthy', 'confidence': 100, 'note': 'No face detected — appears normal.'}]

        landmarks = results.multi_face_landmarks[0].landmark
        face_width = abs(landmarks[234].x - landmarks[454].x)
        face_height = abs(landmarks[10].y - landmarks[152].y)
        ratio = face_height / (face_width + 1e-5)

        suggestions = []
        if ratio < 1.1:
            suggestions.append({'syndrome': 'Possible Down Syndrome', 'confidence': 55, 'note': 'Broad facial ratio detected.'})
        elif ratio > 1.9:
            suggestions.append({'syndrome': 'Possible Marfan Syndrome', 'confidence': 60, 'note': 'Elongated facial structure detected.'})

        if not suggestions:
            suggestions = [{'syndrome': 'Healthy', 'confidence': 100, 'note': 'No abnormal facial features detected.'}]
        else:
            max_conf = max(s['confidence'] for s in suggestions)
            if max_conf < 30:
                suggestions = [{'syndrome': 'Healthy', 'confidence': 95, 'note': 'Facial features appear normal.'}]

        return suggestions