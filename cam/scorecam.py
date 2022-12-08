import torch
import torch.nn.functional as F
from base import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(self, model, target_layer_path, device):
        super().__init__(model, target_layer_path, device)

    def forward(self, input, class_idx=None):
        b, c, h, w = input.size()
        input = input.to(self.device)
        # predication on raw input
        logit, _ = self.model(input)

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
            print(score)
        else:
            predicted_class = torch.LongTensor([class_idx])
            print(logit.size())
            score = logit[:, class_idx].squeeze()
            print(score)

        logit = F.softmax(logit, dim=1)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        self.model.zero_grad()
        score.backward(retain_graph=True)
        activations = self.activations
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

              if saliency_map.max() == saliency_map.min():
                continue

              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # how much increase if keeping the highlighted region
              # predication on masked input
              output, _ = self.model(input * norm_saliency_map)
              output = F.softmax(output, dim=1)
              score = output[0][predicted_class]

              score_saliency_map +=  score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map.cpu()[0,0]

    def __call__(self, input, class_idx=None):
        return self.forward(input, class_idx)
