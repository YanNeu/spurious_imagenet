import torch
from captum._utils.common import _format_output
from captum._utils.gradient import _forward_layer_eval

def activations_with_grad(inputs, model, last_layer, attribute_to_layer_input=True, grad_enabled=False):
    if len(inputs.shape) == 3:
        inputs = inputs[None, :]

    layer_eval = _forward_layer_eval(
                model,
                inputs,
                last_layer,
                None,
                attribute_to_layer_input=attribute_to_layer_input,
                grad_enabled=grad_enabled
            )
    if isinstance(last_layer, torch.nn.Module):
        return _format_output(len(layer_eval) > 1, layer_eval)
    else:
        return [
            _format_output(len(single_layer_eval) > 1, single_layer_eval)
            for single_layer_eval in layer_eval
        ]