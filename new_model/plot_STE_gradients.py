#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

class RelaxedRoundingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale=1.0):
        ctx.save_for_backward(input+0.5)
        ctx.scale = scale
        # Soft rounding uses torch.round in forward
        rounded = torch.round(input)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        scale = ctx.scale
        # Compute sigmoid approximation gradient
        sigmoid_approx = torch.sigmoid(scale * (input - torch.round(input)))
        grad_input = grad_output * sigmoid_approx * (1 - sigmoid_approx) * scale
        # for i in range(input.shape[0]): # Flat edges slow
        #     # print(input.shape[0])
        #     if input[i] < 0.5:
        #         grad_input[i]=grad_input[i]*0.0
        #     if input[i] > 3.5:
        #         grad_input[i]=grad_input[i]*0.0
        # grad_input = grad_input*torch.clamp(input,-0.5,1.0)
        l = torch.heaviside(input-0.5,grad_input)
        u = (torch.heaviside(input-3.5,grad_input)-1)*-1
        grad_input = grad_input*u*l
        return grad_input, None

# Generate input values
x = torch.linspace(-0.49, 3.49, 500, requires_grad=True)
scale = 20.0  # Example scale value

# Forward pass
output = RelaxedRoundingFunction.apply(x, scale)

# Backward pass
output.sum().backward()

grad = x.grad.detach().numpy()

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x.detach().numpy(), grad, label=f'Gradient (scale={scale})', color='blue')
plt.plot(x.detach().numpy(), torch.round(x).detach().numpy(), label=f'Forward pass', color='black')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
# plt.title("Gradient of Relaxed Rounding Function")
plt.xlabel("Input")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('STE.pdf')
# %%
