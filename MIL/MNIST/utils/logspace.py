import torch

class SafeLogAddExp(torch.autograd.Function):
    """Implements a torch function that is exactly like logaddexp,
    but is willing to zero out nans on the backward pass."""

    @staticmethod
    def forward(ctx, input, other):
        with torch.enable_grad():
            output = torch.logaddexp(input, other) # internal copy of output
        ctx.save_for_backward(input, other, output)
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, other, output = ctx.saved_tensors
        grad_input, grad_other = torch.autograd.grad(output, (input, other), grad_output, only_inputs=True)
        mask = torch.isinf(input).logical_and(input == other)
        grad_input[mask] = 0
        grad_other[mask] = 0
        return grad_input, grad_other

logaddexp = SafeLogAddExp.apply

def log1mexp(x):
    assert(torch.all(x >= 0))
    return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))
