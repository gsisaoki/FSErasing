import torch
import torch.nn.functional as F


class ParsingCAM():
    def __init__(self, model, seg_model, n_class, device):
        self.model = model
        self.seg_model = seg_model
        self.n_class = n_class
        self.device = device

        self.model.eval()
        self.seg_model.eval()

        self.org_x = None
        self.org_x_blur = None
        self.blur_mask = None
        self.mask = None
        self.mask_x = None

    def __call__(self, x, class_idx=None):
        b, c, h, w = x.size()
        self.org_x = x.clone().detach().cpu() / 2 + 0.5

        with torch.no_grad():
            if self.num_seg_out == 1:
                masks = self.seg_model(x.to(device))
            else:
                masks = self.seg_model(x.to(device))[0]

            masks = masks.softmax(dim=1).argmax(dim=1)
            self.mask = masks.clone().detach().cpu()

            b_masks = torch.zeros((b, self.num_classes, h, w), dtype=torch.float32)
            for i in range(self.n_class):
                b_masks[:, i, :, :] = torch.where(masks == i, 1.0, 0.0)
            b_masks = b_masks.reshape(-1, 1, h, w)

            self.b_mask = b_masks

            org_b_masks = b_masks.clone()
            org_b_masks = org_b_masks.detach().cpu().reshape(b, -1, h, w)

            x_exp = x.repeat(1, self.n_class, 1, 1).reshape(-1, c, h, w)
            mask_x = x_exp * b_masks

            b_masks = b_masks.detach().cpu().reshape(b, -1, h, w)
            self.mask_x = mask_x.clone().detach().cpu() / 2 + 0.5

            mask_x = torch.cat([mask_x, x], dim=0)

            outputs, _ = self.model(mask_x.to(device))

        outputs = F.softmax(outputs, dim=1).detach().cpu()
        out_m = outputs[:b * self.num_classes]
        out_x = outputs[b * self.num_classes:]

        if class_idx is None:
            class_idx = out_x.max(1)[-1]
        else:
            class_idx = torch.tensor(class_idx)
        class_idx_exp = class_idx.unsqueeze(1).repeat(1, self.num_classes).reshape(-1)

        score = torch.zeros(b * self.num_classes)
        for i in range(len(out_m)):
            score[i] = out_m[i, class_idx_exp[i]]
        score = score.reshape(b, self.num_classes)

        score = F.relu(score).unsqueeze(-1).unsqueeze(-1)
        score_saliency_map = torch.sum(org_b_masks * score, dim=1).reshape(b, -1)

        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(1)[0].unsqueeze(1), score_saliency_map.max(1)[0].unsqueeze(1)

        score_saliency_map = (score_saliency_map - score_saliency_map_min) / (score_saliency_map_max - score_saliency_map_min)
        score_saliency_map = score_saliency_map.reshape(b, h, w)

        return score_saliency_map
