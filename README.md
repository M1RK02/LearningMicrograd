# Learning Micrograd

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

> _"The best way to learn something is to build it from scratch."_

This repository contains my documented and commented implementation of micrograd, the tiny scalar-valued autograd engine originally created by [Andrej Karpathy](https://github.com/karpathy).

The goal of this project is to better understand the "black box" of Deep Learning by mathematically reconstructing the backpropagation algorithm.

## Learning Objectives

- **The Computational Graph:** How tensors (in this case, scalars) track operations to build a DAG (Directed Acyclic Graph).
- **Backpropagation:** Implementing the `.backward()` method and recursively applying the _Chain Rule_ to calculate gradients.
- **Neural Network Primitives:** Building `Neuron`, `Layer`, and `MLP` classes manually without high-level frameworks like PyTorch.

## Resources

- **Original Project**: [karpathy/micrograd](https://github.com/karpathy/micrograd)
- **Lecture**: [Building micrograd by Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)

## Usage Example

```python
from micrograd.autograd import Value

# 1. Build the graph
a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3

# 2. Add some complexity
c += c + 1
d += d * 2 + (b + a).relu()

# 3. Calculate loss
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f

print(f'Output: {g.data:.4f}')

# 4. Backward Pass (Backpropagation)
g.backward()

print(f'Gradient (dg/da): {a.grad:.4f}')
print(f'Gradient (dg/db): {b.grad:.4f}')
```

## Code Structure

- `micrograd/autograd.py`: The core, it implements the `Value` class, handling scalar operations (+, -, \*, /, pow, exp) and their local derivatives.
- `micrograd/layers.py`: A small neural network library built on top of the core (Neurons, Layers, MLPs).
- `notebooks/`: Jupyter notebooks for visualization and training experiments.

## Contact

If you have contributions, need support, have suggestions, or just want to get in touch with me, send me an [email](mailto:picamirko02@gmail.com)!

## License

This software is licensed under the terms of the MIT license.
See [LICENSE](LICENSE) for more details.
