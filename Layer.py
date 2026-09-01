class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
