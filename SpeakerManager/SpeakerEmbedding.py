

import onnx
import onnxruntime


class SpeakerEmbedder :
    def __init__(self, path,device):
        # providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = onnxruntime.InferenceSession(path,
            providers=["CUDAExecutionProvider"]
        )

    def embed(self,x):
        ort_inputs = {self.sess.get_inputs()[0].name: x}
        feat = self.sess.run(None, ort_inputs)[0]

        return feat

