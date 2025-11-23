import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import torch
import torchvision.transforms.functional as F
import numpy as np

from tqdm import tqdm
import cv2
from SpeakerManager.VVAD.VVAD import VVAD_helper
from SpeakerManager.VVAD.hparams import HParam

device = "cpu"
hp_vvad = HParam(
  os.path.join(os.path.dirname(__file__),'SpeakerManager','VVAD','config','v7.yaml'),
  os.path.join(os.path.dirname(__file__),'SpeakerManager','VVAD','config','default.yaml')
)
model = VVAD_helper(hp_vvad)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'SpeakerManager','VVAD','chkpt','v7.pt'), map_location=device))
model.eval()

for idx_face in range(4) : 
    path_vid = f"/home/data/IITP/tmp/face_{idx_face}.avi"
    cap = cv2.VideoCapture(path_vid)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_face = [0,0,0,0]

    out = cv2.VideoWriter(f"/home/data/IITP/tmp/face_{idx_face}_vvad.avi", fourcc, fps,(width, height) )

    frames = []
    for _ in tqdm(range(n_frame)):
        ret, frame = cap.read()
        # frame : (H,W,3)
        if not ret:
            break
        frames.append(frame)
    cap.release()

    pt_frames = np.array(frames)
    print(pt_frames.shape)
    # BGR to RGB
    pt_frames = pt_frames[..., ::-1]
    pt_frames = torch.from_numpy(pt_frames.copy())
    pt_frames = pt_frames.to(torch.float32)/255.0
    pt_frames= pt_frames.permute(0, 3, 1, 2)
    pt_frames = F.rgb_to_grayscale(pt_frames)
    pt_frames = torch.squeeze(pt_frames,1)
    pt_frames = torch.unsqueeze(pt_frames,0)
    print(pt_frames.shape)

    labels,h = model(pt_frames.to(device), timestep=pt_frames.shape[1])
    print(labels.shape)

    for i in range(labels.shape[1]) :
        prob = labels[0,i].item()
        frame = frames[i]
        if prob > 0.7:
            cv2.putText(frame, f"{prob:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else :
            cv2.putText(frame, f"{prob:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        out.write(frame)


    #out.write(frame)
    out.release()

