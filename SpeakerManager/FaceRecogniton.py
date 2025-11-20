from deepface import DeepFace

class FaceRecognizer:
    def __init__(self, 
                 model_name='DeepFace', 
                 distance_metric='cosine',
                 detection_backend='yolov8'
                 ):
        self.model_name = model_name
        self.detection_backend = detection_backend
        self.distance_metric = distance_metric

    def extarct(self, frame):
        results = DeepFace.represent(
            frame, 
            enforce_detection=True,
            model_name=self.model_name,
            detector_backend = self.detection_backend)
        """
        results{'embedding', 'facial_area', 'face_confidence'}
        """

        return results