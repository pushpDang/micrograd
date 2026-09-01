import math
class Value:
  def __init__(self, data, children=(), operand='', label=''):
    self.data = data
    self._prev = set(children)
    self._backward = lambda:None
    self.operand = operand
    self.label = label
    self.grad = 0
  def __repr__(self):
    return f'Value(data={self.data},label={self.label})'
  def __add__(self, other):
    if isinstance(other, Value):
      out = Value(self.data + other.data, (self, other), '+')
    else:
      out = Value(self.data + other, (self,), '+')
    def _backward():
      if isinstance(other, Value):
        self.grad += 1 * out.grad
        other.grad += 1 * out.grad
      else:
        self.grad += 1 * out.grad
    out._backward = _backward
    return out
  def __mul__(self, other):
    if isinstance(other, Value):
      out = Value(self.data * other.data, (self, other), '*')
    else:
      out = Value(self.data * other, (self,), '*')
    def _backward():
      if isinstance(other, Value):
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
      else:
        self.grad += other * out.grad
    out._backward = _backward
    return out
  def backward(self):
    topo = []
    visited = set()
    def build_topo(node):
      if node not in visited:
        visited.add(node)
        for child in node._prev:
          build_topo(child)
        topo.append(node)
    build_topo(self)
    self.grad = 1
    for node in reversed(topo):
      node._backward()
  def tanh(self):
    out = Value(math.tanh(self.data), (self,), 'tanh')
    def _backward():
      self.grad += (1 - out.data**2) * out.grad
    out._backward = _backward
    return out
  def __sub__(self, other):
    if isinstance(other, Value):
      out = Value(self.data - other.data, (self, other), '-')
    else:
      out = Value(self.data - other, (self,), '-')
    def _backward():
      if isinstance(other, Value):
        self.grad += 1 * out.grad
        other.grad += -1 * out.grad
      else:
        self.grad += 1 * out.grad
    out._backward = _backward
    return out
  def __pow__(self, num):
    out = Value(self.data**num, (self,), '**')
    def _backward():
      self.grad += num * (self.data**(num-1))*out.grad
    out._backward = _backward
    return out
  def __radd__(self, other):
    return self + other

  def __rmul__(self, other):
    return self * other
