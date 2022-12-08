import torch
import os


class ParseErasing(object):
    def __init__(self, p=1.0, n_class=25, n_erase=1, value='random', mask_path='./masks.pth', model=None, dataloader=None):
        self.p = p
        self.n_class = n_class
        self.n_erase = n_erase
        self.v = value

        if not(os.path.exists(mask_path)):
            if self.model is not None:
                self._estimate_masks(mask_path, model, dataset)
            else:
                raise Exception("ArgumentError: file provided by 'mask_path' does not exist, so it is necessary to provide 'model' and 'dataloader' to estimate new masks.") 

        self.masks = torch.load(mask_path)

        self.h = self.masks.size(1)
        self.w = self.masks.size(2)
        self.cls_prob = torch.ones(self.n_class, dtype=torch.float32)


    def _estimate_masks(self):
        self.model.eval()
        
        masks = []
        for img in dataloader:
        


    def __call__(self, img, index, flip):
        if self.masks is not None:
            self.mask = self.masks[index]
        else:
            self.mask = torch.from_numpy(cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE))
        self.mask = multi2binary(self.mask.unsqueeze(0), num_classes=self.n_class)[0].bool()

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
                elif self.v == 'choice_all_1':
                    y, x = torch.randint(0, self.h, (2,))
                    v = torch.empty([3, self.h, self.w], dtype=torch.float32)
                    v[:,:,:] = img[:, y, x].unsqueeze(1).unsqueeze(1)
                    img = torch.where(self.mask[cls_id], v, img)
                elif self.v == 'choice_mask_1':
                    yx = torch.nonzero(self.mask[cls_id])
                    if len(yx) == 0:
                        y, x = torch.randint(0, self.h, (2,))
                    else:
                        idx = torch.randint(0, len(yx), (1,))[0]
                        y, x = yx[idx]
                    v = torch.empty([3, self.h, self.w], dtype=torch.float32)
                    v[:,:,:] = img[:, y, x].unsqueeze(1).unsqueeze(1)
                    img = torch.where(self.mask[cls_id], v, img)
                elif self.v == 'mean_all_1':
                    avg = torch.mean(img, dim=(1,2))
                    v = torch.empty([3, self.h, self.w], dtype=torch.float32)
                    v[:,:,:] = avg.unsqueeze(1).unsqueeze(1)
                    img = torch.where(self.mask[cls_id], v, img)
                elif self.v == 'mean_mask_1':
                    y, x = torch.nonzero(self.mask[cls_id], as_tuple=True)
                    if len(y) == 0:
                        avg = torch.mean(img, dim=(1,2))
                    else:
                        avg = torch.mean(img[:, y, x], dim=(1,))
                    v = torch.empty([3, self.h, self.w], dtype=torch.float32)
                    v[:,:,:] = avg.unsqueeze(1).unsqueeze(1)
                    img = torch.where(self.mask[cls_id], v, img)
                elif self.v == 'choice_all':
                    y, x = torch.nonzero(self.mask[cls_id], as_tuple=True)
                    f_img = img.flatten(start_dim=1)
                    idx = torch.randint(0, f_img.size(1), (len(y),))
                    v = f_img[:, idx]
                    img[:, y, x] = v
                elif self.v == 'choice_mask':
                    yx = torch.nonzero(self.mask[cls_id])
                    l_yx = len(yx)
                    if l_yx == 0:
                        f_img = img.flatten(start_dim=1)
                        idx = torch.randint(0, f_img.size(1), (len(yx),))
                        v = f_img[:, idx]
                    else:
                        idx = torch.randint(0, l_yx, (l_yx,))
                        t_yx = yx[idx, :]
                        v = img[:, t_yx[:, 0], t_yx[:, 1]]
                    img[:, yx[:, 0], yx[:, 1]] = v
                else:
                    v = torch.tensor(self.v).float()
                    img = torch.where(self.mask[cls_id], v, img)

        return img
