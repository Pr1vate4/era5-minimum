import torch
ckpt = torch.load('/Users/vsevolod/.cache/era5-minimum/cra5/cra5_159v_150k.pth', map_location='cpu', weights_only=True)
for k in ckpt.keys():
    if 'pos_embed' in k:
        print(k)
