import numpy as np

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

class Speaker :
    def __init__(self, speaker_id, embedding, pos):
        self.id = speaker_id

        # embeddings
        self.face_embedding = embedding
        self.speech_embedding = None

        # Tick-related
        self.last_position = pos
        self.is_detected = False
        self.last_timestamp = -1
        self.list_face = []
        self.list_pos = []
        self.list_vvad = []

    def update_speech_embedding(self, embedding,alpha = 0.1):
        if self.speech_embedding is None :
            self.speech_embedding = embedding
        else :
            self.speech_embedding = (1-alpha)*self.speech_embedding + alpha*embedding
        return

    def compare_speech(self, embedding,threshold = 0.5):
        if self.speech_embedding is None :
            return False

        symilarity = cosine_similarity(self.speech_embedding, embedding)

        if symilarity > threshold :
            return True
        else :
            return False

    ### Face Related Functions ###
    def insert_face(self,face):
        self.list_face.append(face)
        self.face_continuity = True

    def compare_face(self, embedding,threshold = 0.5):
        symilarity = cosine_similarity(self.face_embedding, embedding)

        if symilarity > threshold :
            return True
        else :
            return False

    def update_face_embedding(self, embedding,alpha = 0.1):
        if self.speech_embedding is None :
            self.speech_embedding = embedding
        else : 
            self.face_embedding = (1-alpha)*self.face_embedding + alpha*embedding
        return

    def tracking(self, position, threshold = 50):
        dist = np.linalg.norm(np.array(self.last_position[:2]) - np.array(position[:2]))

        if dist < threshold :
            self.last_position = position
            return True
        else :
            return False

    def GetFaces(self):
        return self.list_face

    ## Continuity Functions ##
    def Tick(self,timestamp):
        if self.last_timestamp == -1 :
            self.last_timestamp = timestamp

        if not self.is_detected :
            self.list_face.append(None)
        
        self.is_detected = False

    def Release(self):
        self.last_timestamp = -1
        self.list_face = []
        self.list_pos = []
        self.list_vvad = []







    


