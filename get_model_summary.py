import torch
from models import Wav2Lip
from torchsummary import summary
import numpy as np
import torch.onnx

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


model = load_model("checkpoints/wav2lip_gan.pth")
# model = Wav2Lip().to(device)
print(summary(model, [(1, 512, 1, 1), (512, 512, 1, 1)]))

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
