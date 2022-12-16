import os
import torch
from tqdm import tqdm


def multi2binary(masks, n_class):
    b, h, w = masks.size()
    b_masks = torch.zeros((b, n_class, h, w), dtype=torch.long)

    for i in range(n_class):
        b_masks[:, i, :, :] = torch.where(masks == i, 1.0, 0.0)

    return b_masks



class FSErasing(object):
    def __init__(self, p=1.0, n_class=25, n_erase=1, value='random', mask_path='./masks.pth', model=None, dataloader=None, device=None):
        self.p = p
        self.n_class = n_class
        self.n_erase = n_erase
        self.v = value

        if not(os.path.exists(mask_path)):
            if self.model is not None:
                self._estimate_masks(mask_path, model, dataloader, device)
            else:
                raise Exception("ArgumentError: file provided by 'mask_path' does not exist, so it is necessary to provide 'model' and 'dataloader' to estimate new masks.") 

        self.masks = torch.load(mask_path)

        self.h = self.masks.size(1)
        self.w = self.masks.size(2)
        self.cls_prob = torch.ones(self.n_class, dtype=torch.float32)


    def _estimate_masks(self, mask_path, model, dataloader, device):
        if device is None:
            device = 'cpu'

        model = torch.nn.DataParallel(model).to(device)
        model.eval()
        
        masks = []
        for inputs in dataloader:
            inputs = inputs.to(device)
            mask = model(inputs)
            mask = mask.softmax(dim=1).cpu().detach().to(torch.uint8)
            masks.append(mask)
            
        masks = torch.cat(masks, dim=0)
        
        torch.save(masks, mask_path)


    def __call__(self, img, index, flip=False):
        self.mask = multi2binary(self.masks[index].unsqueeze(0), n_class=self.n_class)[0].bool()

        if flip:
            self.mask = torch.flip(self.mask, (2,))

        if torch.rand(1) < self.p:
            if type(self.n_erase) is tuple:
                n_target = torch.randint(self.n_erase[0], self.n_erase[1] + 1, (1,))[0]
                self.target_classes = self.cls_prob.multinomial(num_samples=n_target, replacement=False)
            elif type(self.n_erase) is list:
                self.target_classes = self.n_erase
            else:
                n_target = self.n_erase
                self.target_classes = self.cls_prob.multinomial(num_samples=n_target, replacement=False)

            for cls_id in self.target_classes:
                if self.v == 'random':
                    v = torch.empty([3, self.h, self.w], dtype=torch.float32).normal_()
                    img = torch.where(self.mask[cls_id], v, img)
                elif self.v == 'random_1':
                    color = torch.normal(mean=0, std=1, size=(3,1,1))
                    v = torch.empty([3, self.h, self.w], dtype=torch.float32)
                    v[:,:,:] = color
                    img = torch.where(self.mask[cls_id], v, img)
                elif self.v == 'random_mix':
                    if torch.rand(1) < 0.5:
                        v = torch.empty([3, self.h, self.w], dtype=torch.float32).normal_()
                    else:
                        color = torch.normal(mean=0, std=1, size=(3,1,1))
                        v = torch.empty([3, self.h, self.w], dtype=torch.float32)
                        v[:,:,:] = color
                    img = torch.where(self.mask[cls_id], v, img)
                elif type(self.v) in [int, float]:
                    v = torch.tensor(self.v).float()
                    img = torch.where(self.mask[cls_id], v, img)

        return img
