"""Research-only objective adapter; frozen GMPI architecture/trainer files stay intact."""
from __future__ import annotations

import ast
import inspect
from textwrap import dedent


def query_cross_entropy(probabilities, labels, valid_counts, temperature=0.1):
    """Cross entropy on already-normalized GMPI outputs, not logits."""
    import torch
    require = lambda condition, message: _require(bool(condition), message)
    require(probabilities.ndim == 3 and probabilities.shape[-1] == 1
            and probabilities.shape == labels.shape, "Expected batch/candidate/singleton tensors")
    require(temperature > 0, "Positive label temperature required")
    count = torch.as_tensor(valid_counts, device=probabilities.device).reshape(-1)
    require(len(count) == len(probabilities) and torch.all(count > 0), "Invalid query candidate counts")
    count = count.clamp(max=probabilities.shape[1])
    positions = torch.arange(probabilities.shape[1], device=probabilities.device)
    mask = positions[None, :] < count[:, None]
    p, y = probabilities.squeeze(-1), labels.detach().squeeze(-1)
    require(torch.isfinite(p).all() and torch.isfinite(y).all(), "Nonfinite probabilities or labels")
    require(torch.all((p >= 0) & (p <= 1)), "Expected normalized probabilities")
    require(torch.all((y[mask] >= 0) & (y[mask] <= 1)), "Invalid continuous labels")
    require(torch.allclose(p.masked_fill(~mask, 0).sum(dim=1), torch.ones_like(p[:, 0]),
                           rtol=1e-4, atol=1e-5), "GMPI probability mass differs from valid candidates")
    target = torch.softmax((y / temperature).masked_fill(~mask, float('-inf')), dim=1)
    logp = p.clamp_min(torch.finfo(p.dtype).tiny).log().masked_fill(~mask, 0)
    return -(target * logp).sum(dim=1).mean()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def replace_training_loss(function, temperature=0.1):
    """Change exactly the one original training loss call; no global Torch patch."""
    tree = ast.parse(dedent(inspect.getsource(function)))
    changed = []
    class Replace(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'F' and node.func.attr == 'smooth_l1_loss'):
                _require(len(node.args) == 2 and not node.keywords
                         and isinstance(node.args[0], ast.Name) and node.args[0].id == 'pathScore'
                         and ast.dump(node.args[1]) == ast.dump(ast.parse('similarity.detach()', mode='eval').body),
                         "Unexpected frozen training loss signature")
                changed.append(node)
                return ast.copy_location(ast.Call(func=ast.Name(id='_research_query_loss', ctx=ast.Load()),
                    args=[*node.args, ast.parse('item[9]', mode='eval').body,
                          ast.Constant(temperature)], keywords=[]), node)
            return node
    tree = Replace().visit(tree)
    _require(len(changed) == 1, "Expected exactly one frozen Smooth L1 training call")
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__, _research_query_loss=query_cross_entropy)
    exec(compile(tree, '<research GMPI training loss only>', 'exec'), namespace)
    return namespace[function.__name__]
