import torch
import torch.nn as nn
import SpeakerManager.VVAD._coder as _coder
from torchvision import transforms

class TGRU(nn.Module):
    def __init__(self,
        n_feature,
        dropout=0.0,
        **kwargs
        ) :
        super(TGRU,self).__init__()

        self.gru = nn.GRU(n_feature,n_feature,bidirectional=True,batch_first = True,dropout=dropout)
        self.f_glu = nn.GLU()
        self.f_norm = nn.InstanceNorm1d(n_feature)
        self.b_glu = nn.GLU()  
        self.b_norm = nn.InstanceNorm1d(n_feature)

    def forward(self,x,h = None) : 
        y = torch.permute(x,(0,2,1))
        y,h = self.gru(y,h)
        y = torch.permute(y,(0,2,1))

        fv = y[:,:y.shape[1]//2,:]
        bv = y[:,y.shape[1]//2:,:]

        fv = nn.functional.relu(self.f_norm(fv))
        bv = nn.functional.relu(self.b_norm(bv))
        y = x + fv + bv

        return y,h

class FSA(nn.Module):
    def __init__(self,n_channels, num_heads=4,dropout=0.0) :
        super(FSA,self).__init__()

        self.SA = nn.MultiheadAttention(n_channels, n_channels, batch_first = True,dropout=dropout)
        self.bnsa = nn.BatchNorm1d(n_channels)
        self.relu = nn.ReLU()


    def forward(self,x) :
        # x : [B,C,T] -> [B,T,C]
        x_ = torch.permute(x,(0,2,1))

        ysa,h = self.SA(x_,x_,x_)
        ysa = torch.permute(ysa,(0,2,1))
        ysa = self.bnsa(ysa)
        ysa = self.relu(ysa)

        output = x + ysa
        return output

class Labeler(nn.Module):
    def __init__(self,
        n_feature,
        **kwargs) : 
        super(Labeler,self).__init__()

        self.fc = nn.Linear(n_feature,1)
        self.activation = nn.Sigmoid()

    def forward(self,x) : 
        y = torch.permute(x,(0,2,1))
        y = self.fc(y)
        y = self.activation(y)
        return y

class VVAD(nn.Module):
    def __init__(self, hp) : 
        super(VVAD,self).__init__()

        arch = hp.model.architecture

        encoder = getattr(_coder,hp.model.encoder)

        # encoder
        self.enc = []

        for i in range(len(arch["encoder"])) : 
            module = encoder(**arch["encoder"][f"enc{i+1}"])
            self.add_module(f"enc{i+1}",module)
            self.enc.append(module)

        # Bottleneck
        ## Temporal
        self.TemporalBottleneck= TGRU(**arch["TB"])

        ## Feature
        #self.FrequencialBottleneck = FSA(256, num_heads=4, dropout=0.0)

        ## Channel

        ## Label
        self.labeler = Labeler(**arch["labeler"])

        self.enc = nn.ModuleList(self.enc)

    def forward(self,x,h=None) : 
        """
            x : (B, 1, 96,96, T)
            40ms per each timestep
        """
        # encoder
        for enc in self.enc : 
            x = enc(x)
            #print("enc : {} | {}".format(x.shape, x.shape[1]*x.shape[2]*x.shape[3]))

        x = torch.squeeze(x,2)
        x = torch.squeeze(x,2)

        #print("Bottleneck : {}".format(x.shape))
        # bottlneck

        x,h = self.TemporalBottleneck(x)

        #x = self.FrequencialBottleneck(x)

        y = self.labeler(x)
        y = torch.permute(y,(0,2,1))
        return y,h

class VVAD_helper(nn.Module) : 
    def __init__(self,hp,device="cpu") : 
        super(VVAD_helper,self).__init__()
        self.m = VVAD(hp)
        self.device = device
        self.min_t = 16

    def forward(self,x,timestep=38, h= None) : 
        # x : (B, T, 96,96) -> (B,1,96,96,T)
        x = torch.permute(x,(0,2,3,1))
        x = torch.unsqueeze(x,1)
        y,h = self.m(x,h)
        # y : (B,1,T',1) -> (B,T,1)
        y = nn.functional.interpolate(y,(timestep),mode='linear')
        y = torch.squeeze(y,1)

        return y,h
    
    def Resolve(self,list_faces):

        n_frame = len(list_faces)
        prev_flag = False
        cur_flag = False
        cur_face = []
        probs = []

        with torch.no_grad() :
            for face in list_faces : 
                if face is None :
                    cur_flag = False

                    # End of continuity
                    if prev_flag and len(cur_face)  >= self.min_t:
                        faces = torch.stack(cur_face,dim=1).to(self.device)
                        predict,h = self.forward(faces,timestep=len(cur_face))
                        for p in predict[0,:] :
                            probs.append(p.item())
                        cur_face = []
                    else : 
                        for x in cur_face :
                            probs.append(-1.0)
                        cur_face = []

                    probs.append(-1.0)
                else : 
                    cur_flag = True
                    cur_face.append(face)
                prev_flag = cur_flag

            # TODO : update for continous block
            # Force End
            if len(cur_face) >= self.min_t :
                faces = torch.stack(cur_face,dim=1).to(self.device)
                predict,h = self.forward(faces,timestep=len(cur_face))
                for p in predict[0,:] :
                    probs.append(p.item())
            else :
                # +1 for last frame
                for i in range(len(cur_face)):
                    probs.append(-1.0)

            if len(list_faces) != len(probs) :
                import pdb
                pdb.set_trace()
            return probs

if __name__ == "__main__" : 
    import SpeakerManager.VVAD.hparams as HParam
    import os

    hp = HParam.HParam(
        os.path.join(os.path.dirname(__file__),'config','v12.yaml'),
        os.path.join(os.path.dirname(__file__),'config','default.yaml')
    )

    model = VVAD_helper(hp)
    x = torch.randn(1,38,3,96,96)
    print(x.shape)
    y,h = model(x,timestep=38)
    print(y.shape)