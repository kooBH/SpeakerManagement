import os 
import numpy as np
import torch
import torchvision.transforms.functional as F
import cv2

from SpeakerManager.Speaker import Speaker
from SpeakerManager.FaceRecogniton import FaceRecognizer
from SpeakerManager.VVAD.VVAD import VVAD_helper
from SpeakerManager.VVAD.hparams import HParam

class SpeakerManager:
    def __init__(self, threshold, device='cpu'):
        self.speakers = []
        self.threshold = threshold

        self.timestamp = 0

        # Modules 
        self.FR = FaceRecognizer()
        hp_vvad = HParam(
            os.path.join(os.path.dirname(__file__),'VVAD','config','v7.yaml'),
            os.path.join(os.path.dirname(__file__),'VVAD','config','default.yaml')
        )
        self.VVAD = VVAD_helper(hp_vvad)
        self.VVAD.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'VVAD','chkpt','v7.pt'), map_location=device))
        self.VVAD.eval()

    # TODO : will be 'insert_utterance' function
    def insert_video(self,frames):
        n_frame = len(frames)
        ret = {}
        
        # Face Detection & Face Embedding
        for frame in frames :
            self.insert_image(frame)

            for idx, spk in enumerate(self.speakers):
                spk.Tick(self.timestamp)

            self.timestamp += 1
        
        ret['n_speaker'] = len(self.speakers)
        for idx in range(len(self.speakers)):
            ret[f'spk{idx}'] = {}
            ret[f'spk{idx}']['face_pos'] = self.speakers[idx].list_pos

        # VVAD
        for idx,spk in enumerate(self.speakers):
            probs = self.VVAD.Resolve(spk.list_face)
            ret[f'spk{idx}']['vvad'] = probs


        for idx,spk in enumerate(self.speakers):
            spk.Release()

        """
        return informations about this video
        ret['n_speaker'] = len(self.speakers)
        ret['spk{i}']['face_pos'] : face position / -1 for no Face
        ret['spk{i}']['vvad'] : VVAD prob / -1 for No face

        """
        return ret

    def insert_image(self,image):
        # image : H x W x C
        # Detect & Extract face embedding
        results = self.FR.extarct(image)

        #print(len(results))

        # for each face
        for i_result, result in enumerate(results) :
            embedding = np.array(result['embedding'])
            xs = result['facial_area']['x']
            ys = result['facial_area']['y']
            w = result['facial_area']['w']
            h = result['facial_area']['h']
            position = np.array([xs,ys,w,h])

            #print(position)


            ## Check for duplicated detection
            """
            valid_detection = True

            for j_result in range(i_result) :
                prev_xs = results[j_result]['facial_area']['x']
                prev_ys = results[j_result]['facial_area']['y']
                dist = np.linalg.norm(np.array([xs,ys]) - np.array([prev_xs,prev_ys]))
                if dist < 20 :
                    valid_detection = False
                    break
            if not valid_detection :
                continue
            """

            known_speaker = False

            # face : (H,W,3) -> (96,96)
            face = image[ys:ys+h, xs:xs+w]
            face = cv2.resize(face,(96,96))
            face = torch.from_numpy(face.copy())
            face = face.to(torch.float32)/255.0
            face = F.rgb_to_grayscale(face.permute(2,0,1))

            # first speaker 
            if len(self.speakers) == 0 :
                print(f"First Speaker {0}")
                speaker = Speaker(speaker_id = 0, embedding = embedding, pos = position)
                speaker.insert_face(face,position)
                self.speakers.append(speaker)
                known_speaker = True
                continue
            # update
            else :
                # tracking based on position
                for idx, spk in enumerate(self.speakers):
                    known_speaker = spk.tracking(position)
                    if known_speaker : 
                        spk.insert_face(face,position)
                        break
                
                # face matching
                if not known_speaker :
                    for idx, spk in enumerate(self.speakers):
                        known_speaker = spk.compare_face(embedding, threshold = self.threshold)

                    if known_speaker :
                        try : 
                            spk.update_face_embedding(embedding, alpha = 0.1)
                            spk.insert_face(face)
                        except Exception as e:
                            print(e)
                            import pdb; pdb.set_trace()
                        break
                        #print(f"Speaker ID : {spk.id} updated")

            # new speaker
            if not known_speaker :
                speaker = Speaker(speaker_id = len(self.speakers), embedding = embedding, pos = position)
                speaker.insert_face(face,position)
                self.speakers.append(speaker)
                print(f"New Speaker ID : {len(self.speakers)-1}")

    def insert_utterance(self,video,audio = None):
        # video : []
        # audio : 

        # Face Detection & Face Embedding

        for frame in video :
            results = self.FR.extarct(frame)


        # Face Matching


        # VVAD


        # Speaker embedding 


        # Speaker management
        

        return

    def resolve_VVAD(self,idx):
        return self.VVAD(self.speakers[idx].GetFaces())

    def process(self):
        return

    def reset(self):
        return

