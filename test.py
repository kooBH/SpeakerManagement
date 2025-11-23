import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tqdm import tqdm
import cv2
import numpy as np
import librosa as rs
import scipy.io as sio
from SpeakerManager.Manager import SpeakerManager
from PIL import ImageFont, ImageDraw, Image


# 2 sec unit
unit_video = 60 #  frame
unit_audio = 32000 # sample
unit_doa = 128
ratio_v2a = unit_audio/unit_video/unit_doa

manager = SpeakerManager(threshold = 0.7)

# Video
#path_vid = "/home/data/IITP/dev/try_5_아이스크림_10sec.mp4"
path_vid = "/home/data/IITP/dev/try_5_아이스크림.avi"
last_processed_frame = 0

cap = cv2.VideoCapture(path_vid)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#cv2.putText(t_frames[idx],f"∠{int(angle_est)}°",(x,y+h+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
def PutUnicode(frame, text, pos) :
    font_path = "PretendardVariable.ttf" 
    font = ImageFont.truetype(font_path, 15)

    img_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    draw.text(pos,text,font=font, fill=(255,255,255))
    frame = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)

    return frame

    

def vid2angle(x,y) : 
    """
    485, 79 -> -20
    815,100 -> 20
    840,430 -> -150
    480,432 -> 150
    """

    up_point = np.array([0,485,640,815,1280])
    up_angle = np.array([-90,-20,0,20,90])

    down_point = np.array([0,480,639,640,840,1280])
    down_angle = np.array([90,150,180,-180,-150,-90])

    # upper side : -90 ~  0 ~ 90
    if y < 360 :
        angle = np.interp(x, up_point, up_angle)
    # lower side :  90 ~ 180==-180 ~ -90
    else :  
        angle = np.interp(x, down_point, down_angle)
    return angle


print(f"{path_vid}| FPS: {fps}, Width: {width}, Height: {height}, Number of frames: {n_frame}")


fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
prev_face = [0,0,0,0]


## Audio
path_audio1 = "/home/data/IITP/dev/try_5_icecream_0.wav"
path_audio2 = "/home/data/IITP/dev/try_5_icecream_1.wav"

path_doa1 = "/home/data/IITP/dev/try_5_icecream_0_doa.mat"
path_doa2 = "/home/data/IITP/dev/try_5_icecream_1_doa.mat"

path_vad1 = "/home/data/IITP/dev/try_5_icecream_0_vad.mat"
path_vad2 = "/home/data/IITP/dev/try_5_icecream_1_vad.mat"

audio1 = rs.load(path_audio1,sr=16000)[0]
audio2 = rs.load(path_audio2,sr=16000)[0]

doa1 = sio.loadmat(path_doa1)['doa'][0]
doa2 = sio.loadmat(path_doa2)['doa'][0]

vad1 = sio.loadmat(path_vad1)['vad'][0]
vad2 = sio.loadmat(path_vad2)['vad'][0]

idx_audio1 = 0
idx_audio2 = 0
idx_doa1 = 0 
idx_doa2 = 0

# approximated video frame to audio frame
approx_idx = 0


## Process

out = cv2.VideoWriter("/home/data/IITP/tmp/out.avi", fourcc, fps,(width, height) )


frames = []
t_frames=[]
n_spk = 0
for i_frame in tqdm(range(n_frame)):
    ret, frame = cap.read()

    # frame : (H,W,3)
    if not ret:
        break

    # Debugging
    #if i_frame < 11100 and i_frame > 80 :
    #    continue

    t_frames.append(frame)
    #print(frame.shape)
    #manager.insert_image(frame)

    if len(t_frames) == unit_video :

        list_doa1=[]
        list_doa2=[]

        # Audio
        for idx in range(unit_video) :
            approx_st = int(ratio_v2a*(last_processed_frame+idx))
            approx_ed = int(ratio_v2a*(last_processed_frame+idx+1))
            c_doa1 = int(np.mean(doa1[approx_st:approx_ed]))
            c_doa2 = int(np.mean(doa2[approx_st:approx_ed]))

            c_vad1 = np.mean(vad1[approx_st:approx_ed])
            c_vad2 = np.mean(vad2[approx_st:approx_ed])

            if c_vad1 < 0.5 : 
                c_doa1  = "sil"
            if c_vad2 < 0.5 :
                c_doa2 = "sil"
            list_doa1.append(c_doa1)
            list_doa2.append(c_doa2)
            cv2.putText(t_frames[idx],f"estim doa 1 : {c_doa1}",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            cv2.putText(t_frames[idx],f"estim doa 2 : {c_doa2}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)


        # Video
        info = manager.insert_video(t_frames)
        n_spk = info['n_speaker']

        for i_spk in range(n_spk) : 
            face_pos = info[f'spk{i_spk}']['face_pos']
            #print(f"{i_spk} | {face_pos}")
            vvad = info[f'spk{i_spk}']['vvad']

            if len(face_pos) != len(vvad) :
                import pdb
                pdb.set_trace()

            # align timestamp for newly created Speaker
            if len(face_pos) != len(t_frames) :
                pre_pad = len(t_frames) - len(face_pos)
                face_pos = [None]*pre_pad + face_pos
                vvad = [-1.0]*pre_pad + vvad

            for idx, pos in enumerate(face_pos) :
                if pos is not None :
                    x,y,w,h = pos
                    prob = vvad[idx]
                    
                    cv2.rectangle(t_frames[idx], (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(t_frames[idx], f'SPK{i_spk}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                    angle_est = vid2angle(x,y)
                    c_stream = -1
                    if type(list_doa1[idx]) != str :
                        dist = np.abs(angle_est-list_doa1[idx])
                        if dist < 20 :
                            c_stream = 1

                    if type(list_doa2[idx]) != str :
                        dist = np.abs(angle_est-list_doa2[idx])
                        if dist < 20 :
                            c_stream = 2


                    if c_stream == 1 :
                        color = (255,0,0)
                    else : 
                        color = (128,128,128)
                    cv2.rectangle(t_frames[idx], (x,y+h), (x+w,y+h+20), color, cv2.FILLED)
                    cv2.putText(t_frames[idx],f"S1",(x,y+h+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
                    h+=20

                    if c_stream == 2 :
                        color = (255,0,0)
                    else : 
                        color = (128,128,128)
                    cv2.rectangle(t_frames[idx], (x,y+h), (x+w,y+h+20), color, cv2.FILLED)
                    cv2.putText(t_frames[idx],f"S2",(x,y+h+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
                    h+=20

                    # (G,B,R)
                    if prob > 0.5 :
                        color = (255,0,0)
                    elif prob == -1.0 :
                        color = (128,128,128)
                    else :
                        color = (0,0,255)
                    cv2.rectangle(t_frames[idx], (x,y+h), (x+w,y+h+20), color, cv2.FILLED)
                    cv2.putText(t_frames[idx],f"V{prob:.2f}",(x,y+h+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
                    h+=20
                    color = (128,128,128)


                    cv2.rectangle(t_frames[idx], (x,y+h), (x+w,y+h+20), color, cv2.FILLED)
                    #cv2.putText(t_frames[idx],f"∠{int(angle_est)}°",(x,y+h+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
                    t_frames[idx] = PutUnicode(t_frames[idx],f"θ {int(angle_est)}°",(x,y+h))

            
        for idx in range(len(t_frames)):
            out.write(t_frames[idx])

        last_processed_frame = i_frame
        t_frames=[]
cap.release()
out.release()
