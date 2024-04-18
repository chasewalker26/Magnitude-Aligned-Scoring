import torch

# This file implements IG  and LIG
# For the attributions, the input is a normalized tensor image (1, 3, 224, 224)

def IG(input, model, steps, batch_size, alpha_star, baseline, device, target_class):
    if (steps % batch_size != 0):
        print("steps must be evenly divisible by batch size: " + str(batch_size) + "!")
        return 0, 0, 0, 0

    loops = int(steps / batch_size)

    # generate alpha values as 4D
    alphas = torch.linspace(0, 1, steps, requires_grad = True)

    alphas = alphas.reshape(steps, 1, 1, 1).to(device)

    # array to store the gradient at each step
    gradients = torch.zeros((steps, input.shape[1], input.shape[2], input.shape[3])).to(device)
    # array to store the logit at each step
    logits = torch.zeros(steps).to(device)

    if torch.is_tensor(baseline):
        baseline = baseline
    else:
        baseline = torch.full(input.shape, baseline, dtype = torch.float)

    input = input.to(device)
    baseline = baseline.to(device)
    baseline_diff = torch.sub(input, baseline)

    # run batched input
    for i in range(loops):
        start = i * batch_size
        end = (i + 1) * batch_size

        interp_imgs = torch.add(baseline, torch.mul(alphas[start : end], baseline_diff))

        gradients[start : end], logits[start : end] = getGradientsParallel(interp_imgs, model, target_class)

    max_perc = torch.max(logits)
    cutoff_perc = max_perc * alpha_star

    # IG: sum all the gradients
    if alpha_star == 1:
        grads = gradients.mean(dim = 0)
    # LeftIG: sum the gradients up to the cutoff point and no later
    else:
        cutoff_steps = torch.where(logits > cutoff_perc)[0]

        if len(cutoff_steps) != 0:
            cutoff_step = torch.where(logits > cutoff_perc)[0][0]
        else:
            cutoff_step = 1

        # avoid rare case where no attribution is returned
        if cutoff_step == 0:
            cutoff_step = 1

        grads = (gradients[0 : cutoff_step]).mean(dim = 0)

    # multiply sum by (original image - baseline)
    grads = torch.multiply(grads, baseline_diff[0].unsqueeze(0))
    
    return grads.squeeze()

# returns the gradients from the model for an input
def getGradientsParallel(inputs, model, target_class):
    output = model(inputs)
    scores = output[:, target_class]

    gradients = torch.autograd.grad(scores, inputs, grad_outputs = torch.ones_like(scores))[0]

    return gradients.detach().squeeze(), scores.detach().squeeze()

# returns the logit outputs for a batch of images
def getPredictionParallel(inputs, model, target_class):
    # calculate a prediction
    output = model(inputs)

    scores = output[:, target_class].detach()

    return scores.squeeze()