import torch
from micrograd import Value


def check_val(mg_val, pt_tensor, tol=1e-6):
    """
    Helper function to compare Micrograd Value with PyTorch Tensor.
    Checks both the data (forward pass) and the gradient (backward pass).
    """
    # Check Forward Pass
    assert abs(mg_val.data - pt_tensor.data.item()) < tol, \
        f"Forward mismatch: {mg_val.data} vs {pt_tensor.data.item()}"

    # Check Backward Pass
    mg_grad = mg_val.grad
    pt_grad = pt_tensor.grad.item() if pt_tensor.grad is not None else 0.0
    assert abs(mg_grad - pt_grad) < tol, \
        f"Gradient mismatch: {mg_grad} vs {pt_grad}"


def test_sanity_check():
    """
    Simple test with addition, multiplication, and ReLU.
    """
    # --- Micrograd ---
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    # --- PyTorch ---
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.retain_grad()
    y.backward()
    xpt, ypt = x, y

    # Check results
    check_val(ymg, ypt)  # Check final output and gradient (if y has grad)
    check_val(xmg, xpt)  # Check input gradient


def test_accumulation():
    """
    Specific test to see if gradients accumulate correctly
    when a variable is used multiple times (branching).
    """
    # --- Micrograd ---
    a = Value(3.0)
    b = a + a
    c = b * a
    c.backward()
    amg, cmg = a, c

    # --- PyTorch ---
    a = torch.Tensor([3.0]).double()
    a.requires_grad = True
    b = a + a
    c = b * a
    c.retain_grad()
    c.backward()
    apt, cpt = a, c

    check_val(cmg, cpt)
    check_val(amg, apt)


def test_complex_ops():
    """
    Complex test combining multiple operations to ensure correctness.
    """
    # --- Micrograd ---
    a = Value(-2.0)
    b = Value(3.0)
    d = a * b + b**3
    c = a + b
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    # --- PyTorch ---
    a = torch.Tensor([-2.0]).double()
    b = torch.Tensor([3.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    d = a * b + b**3
    c = a + b
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.retain_grad()
    g.backward()
    apt, bpt, gpt = a, b, g

    check_val(gmg, gpt)
    check_val(amg, apt)
    check_val(bmg, bpt)
