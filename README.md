---
license: other
licence_name: bria-rmbg-1.4
license_link: https://bria.ai/bria-huggingface-model-license-agreement/

tags:
- remove background
- background
- background removal
- Pytorch
- vision
- legal liability

extra_gated_prompt: This model weights by BRIA AI can be obtained after a commercial license is agreed upon. Fill in the form below and we reach out to you.
extra_gated_fields:
  Name: text
  Company/Org name: text
  Org Type (Early/Growth Startup, Enterprise, Academy): text
  Role: text
  Country: text
  Email: text
  By submitting this form, I agree to BRIA’s Privacy policy and Terms & conditions, see links below: checkbox
---

# BRIA Background Removal v1.4 Model Card

RMBG v1.4 is our state-of-the-art background removal model, designed to effectively separate foreground from background in a range of
categories and image types. This model has been trained on a carefully selected dataset, which includes:
general stock images, e-commerce, gaming, and advertising content, making it suitable for various use cases. 
Developed by BRIA AI, RMBG v1.4 is available as an open-source tool for non-commercial use.


![examples](t4.png)

### Model Description

- **Developed by:** [BRIA AI](https://bria.ai/)
- **Model type:** Background Removal 
- **License:** [bria-rmbg-1.4](https://bria.ai/bria-huggingface-model-license-agreement/)
  - The model is open for non-commercial use.
  - Commercial use is subject to a commercial agreement with BRIA. [Contact Us](https://bria.ai/contact-us)

- **Model Description:** BRIA RMBG 1.4 is an saliency segmentation model trained exclusively on a professional-grade dataset.



## Training data
Bria-RMBG model was trained over 12,000 high-quality, high-resolution, manually labeled (pixel-wise accuracy), fully licensed images.
For clarity, we provide our data distribution according to different categories, demonstrating our model’s versatility.

### Distribution of images:

| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Objects only | 45.11% |
| People with objects/animals | 25.24% |
| People only | 17.35% |
| people/objects/animals with text | 8.52% |
| Text only | 2.52% |
| Animals only | 1.89% |

| Category | Distribution |
| -----------------------------------| -----------------------------------------:|
| Photorealistic | 87.70% |
| Non-Photorealistic | 12.30% |


| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Non Solid Background | 52.05% |
| Solid Background | 47.95% 


| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Single main foreground object | 51.42% |
| Multiple objects in the foreground | 48.58% |


## Qualitative Evaluation

![examples](results.png)

- **Inference Time :** 1 sec on Nvidia A10 GPU

## Architecture

The model’s architecture is based on [IS-Net](https://github.com/xuebinqin/DIS). 
Yet, we employ a distinct training scheme and utilize our proprietary data for the training process, enhancing the model's effectiveness.


## Usage

```python
import os
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from models import BriaRMBG

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

# inference 
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
```