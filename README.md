---
license: other
tags:
- background-removal
- Pytorch
- vision
---

# BRIA Background Removal v1.4 Model Card

100% automatically Background removal capability across all categories and image types that capture the variety of the world. 
Built and validated on a comprehensive dataset containing an equal distribution of general stock images, eComm, gaming and ads.



### Model Description

- **Developed by:** BRIA AI
- **Model type:** Background removal image-to-image model
- **License:** [bria-rmbg-1.4](BRIA_License.docx)
- **Model Description:** BRIA RMBG 1.4 is an image-to-image model trained exclusively on a professional-grade dataset. It is designed and built for commercial use, subject to a commercial agreement with BRIA.
- **Resources for more information:** [BRIA AI](https://bria.ai/)


### Get Access
BRIA RMBG 1.4 is available under the BRIA RMBG 1.4 License Agreement. To access the model, please contact us. 
By submitting this form, you agree to BRIA’s [Privacy policy](https://bria.ai/privacy-policy/) and [Terms & conditions](https://bria.ai/terms-and-conditions/).


## Training data
Bria-RMBG model was trained over 12000 high quality, high resolution, fully licensed images.
The training set as well as the validation benchmark if a holistic representation of the commercial world containing a distribution of general stock images, eComm, gaming and ads.

Distribution of images:
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

All images were manualy labeled pixel-wise accuratly. 

## Qualitative Evaluation

![examples](results.png)




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