import os
import numpy as np
from skimage import io
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from briarmbg import BriaRMBG


def example_inference():

    input_size=[1024,1024]
    net=BriaRMBG()

    model_path = "./model.pth"
    im_path = "./example_image.jpg"
    result_path = "."

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
    net.eval()    

    # prepare input
    im = io.imread(im_path)
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=input_size, mode='bilinear').type(torch.uint8)
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

    if torch.cuda.is_available():
        image=image.cuda()
    
    #inference
    result=net(image)

    # post process
    result = torch.squeeze(F.interpolate(result[0][0], size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)

    # save result
    im_name=im_path.split('/')[-1].split('.')[0]
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    cv2.imwrite(os.path.join(result_path, im_name+".png"), im_array)


if __name__ == "__main__":
    example_inference()