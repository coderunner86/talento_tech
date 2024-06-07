from django.shortcuts import render
import numpy as np

def relu(x):
    return np.maximum(0, x)

def single_neuron_network(weights, bias, inputs):
    linear_output = np.dot(weights, inputs) + bias
    activated_output = relu(linear_output)
    return linear_output, activated_output

def home(request):
    context = {}
    if request.method == "POST":
        weights = list(map(float, request.POST.get('weights').split(',')))
        inputs = list(map(float, request.POST.get('inputs').split(',')))
        bias = float(request.POST['bias'])

        linear_output, activated_output = single_neuron_network(weights, bias, inputs)
        
        context = {
            'weights': weights,
            'inputs': inputs,
            'bias': bias,
            'linear_output': linear_output,
            'activated_output': activated_output
        }

    return render(request, 'neural_app/home.html', context)

