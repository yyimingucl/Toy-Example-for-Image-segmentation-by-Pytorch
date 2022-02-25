# your implementation goes here
# Load Packages
from parameters import params
import torch
from torchvision import transforms
import os
from pre_process import load_image
from train import FCNN
import matplotlib.pyplot as plt
config = params()

def display(pred_mask):
    plt.figure(figsize=(15, 15))
    plt.title("Predicate Mask")
    plt.imshow(pred_mask.numpy())
    plt.axis('off')
    plt.show()

def predict(image_path):
    # Griding Image
    image = load_image(os.path.join(config.save_dir,"rgb.png"))
    w = config.input_size[0]
    h = config.input_size[1]
    n_h = int(image.shape[2]/w)+1
    n_v = int(image.shape[1]/h)+1
    
    topil = transforms.ToPILImage()
    totensor = transforms.ToTensor()
    pil = topil(image)
    block_list = []
    for i in range(n_h):
        for j in range(n_v):
            box = (w*i, h*j, w*(i+1), h*(j+1))
            block = pil.crop(box)
            block_list.append(totensor(block))
            
    # Load Model
    model = FCNN(config.num_channels, config.kernel_size, config.paddings)
    model.load_state_dict(torch.load(os.path.join(config.save_dir,"FCNN.pt")))
    model.eval()
    
    # Prediction
    pred_list = []
    a = []
    with torch.no_grad():
        for block in block_list:
            pred = model(block.unsqueeze(0))
            pred = pred.squeeze(0)
            pred_list.append(torch.argmax(pred,dim=0))
            a.append(pred)
            
    
    # jont the predicted blocks
    tmp = []
    for i in range(n_h):
        comb = tuple(pred_list[j] for j in range(i*n_v,(i+1)*n_v))
        tmp.append(torch.cat(comb,dim=0))
    pred_mask = tuple(tmp[j] for j in range(len(tmp)))
    pred_mask = torch.cat(pred_mask,dim=1)
    #pred_mask = topil(pred_mask)
    #pred_mask.save(config.save_dir, "predication_mask.png")
    pred_mask = pred_mask[:image.shape[1],:image.shape[2]]
    display(pred_mask)
    return pred_mask
    
#%%  
if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__)) 
    os.chdir(cur_path)
    path = input('input image path:')
    predict(path) 
    
        
    
            
    
    
    

