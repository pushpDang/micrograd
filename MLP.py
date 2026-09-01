class MLP:
    def __init__(self, nin, nouts):
        for i in range(len(nouts)):
            if i == 0:
                self.layers = [Layer(nin, nouts[i])]
            else:
                self.layers.append(Layer(nouts[i-1], nouts[i]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
