## SurrogateHeaviside definition thresh and sigma learnable
# device = 'cpu'

import torch
from torch import nn

device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device in sglThresh.py", device)

class SurrogateHeaviside(torch.autograd.Function):
    # Activation function with surrogate gradient
    #     sigma = 100.0

    @staticmethod
    def forward(ctx, input, sigma):
        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input, sigma)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, sigma = ctx.saved_tensors
        grad_input = grad_output.clone()
        # approximation of the gradient using sigmoid function
        grad = grad_input * sigma * torch.sigmoid(sigma * input) * torch.sigmoid(-sigma * input)

        grad_sigma = grad_input * input * torch.sigmoid(sigma * input) * torch.sigmoid(-sigma * input)

        return grad, grad_sigma


threshold_fn = SurrogateHeaviside.apply


class ThresholdModel(nn.Module):
    def __init__(self, threshold_fn, t=0.5, sigma=100., nb_classes=10):
        super(ThresholdModel, self).__init__()

        # define nb_classes seuils differents, initialisés à 0.5

        #         self.dense = torch.nn.Linear(10, 10)

        self.thresh = torch.nn.Parameter(t * torch.ones(nb_classes), requires_grad=True)
        self.sigma = torch.nn.Parameter(sigma * torch.ones(nb_classes), requires_grad=True)
        self.threshold_fn = threshold_fn

    def forward(self, x):
        out = self.threshold_fn(x.to(device, dtype=torch.float) - self.thresh.to(device, dtype=torch.float),
                                self.sigma.to(device, dtype=torch.float))
        #         out = out.clamp_(min=0.01, max=0.99)
        # out = self.dense(x.to(device, dtype=torch.float))
        # out = F.sigmoid(out)
        # out = self.threshold_fn(out-F.sigmoid(self.thresh.to(device, dtype=torch.float)))
        return out

    def clamp(self):
        self.thresh.data.clamp_(min=0., max=1.)


def F1_loss_objective(binarized_output, y_true):
    # let's first convert binary vector prob into logits
    #     prob = torch.clamp(prob, 1.e-12, 0.9999999)

    #     average = 'macro'
    average = 'micro'
    epsilon = torch.tensor(1e-12)

    if average == 'micro':
        y_true = torch.flatten(y_true)
        binarized_output = torch.flatten(binarized_output)

    true_positives = torch.sum(y_true * binarized_output, dim=0)
    predicted_positives = torch.sum(binarized_output, dim=0)
    positives = torch.sum(y_true, dim=0)
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (positives + epsilon)

    f1 = 2 * ((precision * recall) / (precision + recall + epsilon))
    #     return precision, recall, f1
    return - f1.mean()


def macro_F1_loss_objective(binarized_output, y_true):
    # let's first convert binary vector prob into logits
    #     prob = torch.clamp(prob, 1.e-12, 0.9999999)

    epsilon = torch.tensor(1e-12)
    true_positives = torch.sum(y_true * binarized_output, dim=0)
    predicted_positives = torch.sum(binarized_output, dim=0)
    positives = torch.sum(y_true, dim=0)
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (positives + epsilon)

    f1 = 2 * ((precision * recall) / (precision + recall + epsilon))
    #     return precision, recall, f1
    return - f1.mean()


def setAcc_loss_objective(binarized_output, y_true):
    # let's first convert binary vector prob into logits
    #     prob = torch.clamp(prob, 1.e-12, 0.9999999)

    #     average = 'macro'
    average = 'micro'
    epsilon = torch.tensor(1e-12)

    if average == 'micro':
        y_true = torch.flatten(y_true)
        binarized_output = torch.flatten(binarized_output)

    true_positives = torch.sum(y_true * binarized_output, dim=0)
    #     return precision, recall, f1
    return - true_positives.mean()


