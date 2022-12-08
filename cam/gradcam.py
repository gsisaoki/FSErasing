import torch
import torch.nn.functional as F


class GradCAM(object):
    def __init__(self, model, target_layer_path, device):
        self.model = model
        self.target_layer_path = target_layer_path
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.device = device

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations = output
            return None

        def loop_hook(module_dict, current_block):
            for pos, module in module_dict:
                module_dict_sub = module._modules.items()
                current_layer = current_block + '/' + pos

                if (len(module_dict_sub) == 1) and (list(module._modules.keys())[0] == 'static_padding') and (list(module._modules.keys())[0] != 'module'):
                    if current_layer == self.target_layer_path:
                        module.register_forward_hook(forward_hook)
                        module.register_backward_hook(backward_hook)
                elif len(module_dict_sub) > 0:
                    loop_hook(module_dict_sub, current_layer)
                else:
                    if current_layer == self.target_layer_path:
                        module.register_forward_hook(forward_hook)
                        module.register_backward_hook(backward_hook)
                        print(current_layer)

        init_block = ''
        loop_hook(self.model._modules.items(), init_block)

    def forward(self, input, class_idx=None):
        input = input.to(self.device)
        b, c, h, w = input.size()

        logit, _ = self.model(input)

        if class_idx is None:
            class_idx = logit.argmax(dim=1)

        one_hot_output = torch.FloatTensor(1, logit.size()[-1]).zero_().to(self.device)
        one_hot_output[0][class_idx] = 1

        self.model.zero_grad()
        logit.backward(gradient=one_hot_output, retain_graph=True)
        gradients = self.gradients
        activations = self.activations
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map.cpu()[0, 0]

    def __call__(self, input, class_idx=None):
        return self.forward(input, class_idx)
