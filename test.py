import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tqdm import tqdm
import cv2
from SpeakerManager.Manager import SpeakerManager


unit_video = 60 #  frame
manager = SpeakerManager(threshold = 0.7)

path_vid = "/home/data/IITP/dev/try_5_아이스크림_10sec.mp4"
cap = cv2.VideoCapture(path_vid)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
prev_face = [0,0,0,0]

out = cv2.VideoWriter("/home/data/IITP/tmp/out.avi", fourcc, fps,(width, height) )


frames = []
t_frames=[]
n_spk = 0
for _ in tqdm(range(n_frame)):
    ret, frame = cap.read()
    # frame : (H,W,3)
    if not ret:
        break

    t_frames.append(frame)
    #print(frame.shape)
    #manager.insert_image(frame)

    if len(t_frames) == unit_video :
        manager.insert_video(t_frames)
        t_frames = []

        # 

        if len(manager.speakers) > n_spk :
            n_spk = len(manager.speakers)

        for spk in manager.speakers :
            xs = spk.pos[0]
            ys = spk.pos[1]
            idx = spk.id
            cv2.putText(frame, f"ID : {idx}", (int(xs), int(ys)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


    #cv2.rectangle(frame, (xs, ys), (xe, ye), (0, 255, 0), 2)
    #out.write(frame)
    frames.append(frame)
cap.release()
    
for frame in frames :
    out.write(frame)
out.release()
