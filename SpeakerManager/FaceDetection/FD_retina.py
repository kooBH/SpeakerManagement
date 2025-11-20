# Old : https://github.com/hhj1897/face_detection based on https://github.com/biubug6/Pytorch_Retinaface
# new : https://github.com/serengil/retinaface
# MIT license

import cv2
import numpy as np
from retina_face import RetinaFacePredictor
import time

class FaceExtractor : 
    def __init__(self,
                 device,
                 #model_name = "resnet50"
                 model_name = "mobilenet0.25"
                 ):
        # Load Models
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
        )

    def extract(self,imag, use_zeros=False, shape_zeros=(96,96)):
        detected_faces = self.face_detector(imag, rgb=False)

        if len(detected_faces) == 0 and use_zeros :
            face = None
        else :
            face = self.crop_face(frame,detected_faces[:,:4])
        return face
                
    def crop_face(self, image, face_boxes,crop_ratio = 0.55): 
        centres = (face_boxes[:, [0, 1]] + face_boxes[:, [2, 3]]) / 2.0
        face_sizes = (face_boxes[:, [3, 2]] - face_boxes[:, [1, 0]]).mean(axis=1)
        enlarged_face_box_sizes = (face_sizes / crop_ratio)[:, np.newaxis].repeat(2, axis=1)
        enlarged_face_boxes = np.zeros_like(face_boxes[:, :4])
        enlarged_face_boxes[:, :2] = np.round(centres - enlarged_face_box_sizes / 2.0)
        enlarged_face_boxes[:, 2:] = np.round(enlarged_face_boxes[:, :2] + enlarged_face_box_sizes) + 1
        enlarged_face_boxes = enlarged_face_boxes.astype(int)
        outer_bounding_box = np.hstack((enlarged_face_boxes[:, :2].min(axis=0),
                                        enlarged_face_boxes[:, 2:].max(axis=0)))
        
        pad_widths = np.zeros(shape=(3, 2), dtype=int)
        if outer_bounding_box[0] < 0:
            pad_widths[1][0] = -outer_bounding_box[0]
        if outer_bounding_box[1] < 0:
            pad_widths[0][0] = -outer_bounding_box[1]
        if outer_bounding_box[2] > image.shape[1]:
            pad_widths[1][1] = outer_bounding_box[2] - image.shape[1]
        if outer_bounding_box[3] > image.shape[0]:
            pad_widths[0][1] = outer_bounding_box[3] - image.shape[0]

        for left, top, right, bottom in enlarged_face_boxes:
            left += pad_widths[1][0]
            top += pad_widths[0][0]
            right += pad_widths[1][0]
            bottom += pad_widths[0][0]

        for i in range(len(enlarged_face_boxes)):
            if enlarged_face_boxes[i][0] < 0:
                enlarged_face_boxes[i][0] = 0
            if enlarged_face_boxes[i][1] < 0:
                enlarged_face_boxes[i][1] = 0
            if enlarged_face_boxes[i][2] < 0:
                enlarged_face_boxes[i][2] = 0
            if enlarged_face_boxes[i][3] < 0:
                enlarged_face_boxes[i][3] = 0

        return enlarged_face_boxes 



if __name__ == "__main__" : 
    path_vid = "ES2002a.Closeup1.clip.mp4"
    #video = VideoFileClip(path_vid)
    # fps = video.fps
    cap = cv2.VideoCapture(path_vid)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extractor = FaceExtractor("cpu")

    print(f"{n_frame} | {width},{height}")

    print("== Initialization Complete ==")

    """
    vid = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vid.append(frame)
    vid = np.array(vid)

    print("== Input Video Loaded ==")

    print("== Face Extraction Complete ==")
    """

    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    out = cv2.VideoWriter("out.avi", fourcc, fps,(width, height) )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    prev_face = [0,0,0,0]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face = extractor.extract(frame, use_zeros=True)
        if face is None : 
            pass
        else :
            # There might be face permutation problem. 
            if len(face)  == 1: 
                xs = face[0][0]
                ys = face[0][1]
                xe = face[0][2]
                ye = face[0][3]
                prev_face = face[0]
            else :
                # If there are multiple faces detected, we need to select the face that is closest to the previous frame.
                min_dist = 100000
                for cur_face in face : 
                    dist = np.linalg.norm(np.array(prev_face) - np.array(cur_face))
                    if dist < min_dist : 
                        min_dist = dist
                        prev_face = cur_face
                        xs = cur_face[0]
                        ys = cur_face[1]
                        xe = cur_face[2]
                        ye = cur_face[3]

            #t_frame = cv2.resize(t_frame, (96, 96))
            cv2.rectangle(frame, (xs, ys), (xe, ye), (0, 255, 0), 2)
        out.write(frame)
        idx += 1
        print(f"{idx}/{n_frame} : {idx/n_frame*100:.2f}%", end="\r")
    cap.release()
    out.release()
