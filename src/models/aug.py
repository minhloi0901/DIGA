import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F1
from torchvision.transforms import ColorJitter
import os
from omegaconf import OmegaConf
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps
import random

class AugCO(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'model', 'augco.yaml')
        cfg = OmegaConf.load(config_path)
        
        self.scale = cfg.get('scale', 0.8)
        self.severity = 2.0
        self.threshold = cfg.get('threshold', 0.5)
        
        self.mean = torch.tensor([128.0, 128.0, 128.0]).view(3, 1, 1)
        self.std = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
        
    
    def unnormalize(self, x):
        return (x * self.std.to(x.device)) + self.mean.to(x.device)
    
    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def autocontrast(self, img, severity):
        return ImageOps.autocontrast(img)

    def equalize(self, img, severity):
        return ImageOps.equalize(img)

    def brightness(self, img, severity):
        enhancer = ImageEnhance.Brightness(img)
        factor = 1 + (severity / 30) * 1.8
        return enhancer.enhance(factor)

    def sharpness(self, img, severity):
        enhancer = ImageEnhance.Sharpness(img)
        factor = 1 + (severity / 30) * 1.8
        return enhancer.enhance(factor)
    
    def randaugment_color_jitter(self, x):
        x_unnorm = self.unnormalize(x)
        x_unnorm = x_unnorm.clamp(0, 255)
        x_uint8 = x_unnorm.byte()
        
        img = F.to_pil_image(x_uint8.cpu())
        transforms = [
            (self.autocontrast, "AutoContrast"),
            (self.equalize, "Equalize"),
            (self.brightness, "Brightness"),
            (self.sharpness, "Sharpness")
        ]

        transform, name = random.choice(transforms)
        # print(f"Applying transform: {name}")
        
        img = transform(img, self.severity)
        img_tensor = F.to_tensor(img).to(x.device) * 255
        
        img_tensor = self.normalize(img_tensor)
        return img_tensor
        
    def center_crop_and_resize(self, x, H, W, H_t, W_t):
        top = (H - H_t) // 2
        left = (W - W_t) // 2
        cropped_x = F.crop(x, top, left, H_t, W_t)  
        # resize_x = F.resize(cropped_x, size=[H, W])
        cropped_x = cropped_x.unsqueeze(0)
        upsampled_x = F1.interpolate(cropped_x, size=(H, W), mode='bilinear', align_corners=False)
        return upsampled_x.squeeze(0)
    
    def pixel_consistency_mask(self, p1, p2):
        pred1 = p1.argmax(dim=1)
        pred2 = p2.argmax(dim=1)
        
        max_prob2, _ = p2.max(dim=1)
        
        same_class = (pred1 == pred2)
        
        class_threshold = torch.zeros(self.num_classes, device=p2.device)
        for c in range(self.num_classes):
            class_mask = (pred2 == c)
            if class_mask.sum() == 0:
                continue
            class_probs = max_prob2[class_mask]
            
            sorted_probs, _ = torch.sort(class_probs, descending=True)
            idx = int(self.threshold * len(sorted_probs)) - 1
            idx = max(idx, 0)
            
            class_threshold[c] = sorted_probs[idx]
        
        thresholds_per_pixel = class_threshold[pred2]
        high_confidence = (max_prob2 > thresholds_per_pixel)
        
        mask = (same_class | high_confidence)

        return mask
    
    def forward(self, batch):
        x, y, shape_, name_ = batch
        B, C, H, W = x.shape
        h2, w2 = int(H * self.scale), int(W * self.scale)
        
        batch_x1, batch_x2 = [], []
        
        #Preprocess
        for i in range(B):
            x1 = x[i]
            batch_x1.append(x1)
            
            x2 = x[i]
            # color jitter
            jitter_x2 = self.randaugment_color_jitter(x2)
            x2 = jitter_x2
            x2 = self.center_crop_and_resize(jitter_x2, H, W, h2, w2)
            batch_x2.append(x2)
    
        batch_x1 = torch.stack(batch_x1, dim=0).detach()
        batch_x1.requires_grad_(False)
        
        batch_x2 = torch.stack(batch_x2, dim=0).detach()
        batch_x2.requires_grad_(True)
            
        _, outputs_x1 = self.model(batch_x1, feat=True)
        _, outputs_x2 = self.model(batch_x2, feat=True)
    
        B_o, C_o, H_o, W_o = outputs_x1.shape
        h1, w1 = int(H_o * self.scale), int(W_o * self.scale)
        
        outputs_x1_cropped = []
        for i in range(B):
            out = outputs_x1[i]
            cropped_out = self.center_crop_and_resize(out, H_o, W_o, h1, w1)
            outputs_x1_cropped.append(cropped_out)
            
        outputs_x1_cropped = torch.stack(outputs_x1_cropped, dim=0)
        
        mask = self.pixel_consistency_mask(outputs_x1_cropped, outputs_x2)
        
        return mask, outputs_x1_cropped, outputs_x2
            
            
            