"""Versioned visual graphs, PyTorch catalog and deterministic model generation."""
import inspect
import json
import math
import re
import ast
import torch
from torch import nn

GROUPS = {
    'Linear & embedding': ['Linear', 'Bilinear', 'Embedding', 'EmbeddingBag', 'Identity'],
    'Convolution': [f'{prefix}{dim}d' for prefix in ('Conv', 'ConvTranspose', 'LazyConv', 'LazyConvTranspose') for dim in (1, 2, 3)],
    'Activation': ['ReLU', 'ReLU6', 'LeakyReLU', 'PReLU', 'RReLU', 'ELU', 'CELU', 'SELU', 'GELU', 'SiLU', 'Mish', 'Sigmoid', 'Tanh', 'Hardtanh', 'Hardsigmoid', 'Hardswish', 'Softplus', 'Softsign', 'Softshrink', 'Hardshrink', 'Tanhshrink', 'Softmax', 'Softmin', 'LogSoftmax', 'GLU', 'Threshold'],
    'Pooling': [f'{prefix}{dim}d' for prefix in ('MaxPool', 'AvgPool', 'AdaptiveMaxPool', 'AdaptiveAvgPool', 'MaxUnpool', 'LPPool', 'FractionalMaxPool') for dim in (1, 2, 3)],
    'Normalization': [f'{prefix}{dim}d' for prefix in ('BatchNorm', 'InstanceNorm', 'LazyBatchNorm', 'LazyInstanceNorm') for dim in (1, 2, 3)] + ['GroupNorm', 'LayerNorm', 'RMSNorm', 'LocalResponseNorm'],
    'Dropout': ['Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout'],
    'Shape & merge': ['Flatten', 'Unflatten', 'Reshape', 'Permute', 'Add', 'Multiply', 'Concat', 'Output'],
    'Padding & resize': [f'{prefix}{dim}d' for prefix in ('ReflectionPad', 'ReplicationPad', 'ZeroPad', 'ConstantPad') for dim in (1, 2, 3)] + ['Upsample', 'PixelShuffle', 'PixelUnshuffle', 'Fold', 'Unfold'],
    'Recurrent & attention': ['RNN', 'LSTM', 'GRU', 'RNNCell', 'LSTMCell', 'GRUCell', 'MultiheadAttention', 'Transformer', 'TransformerEncoderLayer', 'TransformerDecoderLayer'],
    'Distance & loss': ['PairwiseDistance', 'CosineSimilarity'] + [name for name in dir(nn) if name.endswith('Loss')],
}
OPS = {'Reshape': {'shape': [1, -1]}, 'Permute': {'dims': [0, 2, 1]}, 'Add': {}, 'Multiply': {}, 'Concat': {'dim': 1}, 'Output': {}}
DEFAULTS = dict(in_features=4, out_features=8, in_channels=3, out_channels=16, kernel_size=3, num_features=16, num_groups=1, num_channels=16, normalized_shape=[16], num_embeddings=100, embedding_dim=16, output_size=[1, 1], hidden_size=16, input_size=4, embed_dim=16, num_heads=1, d_model=16, nhead=1, padding=0, dim=-1, size=2, threshold=0, value=0, norm_type=2, unflattened_size=[2, 2], scale_factor=2)


def catalog():
    entries = []
    known = set()
    groups = {**GROUPS}
    known.update(x for names in groups.values() for x in names)
    groups['Other modules'] = [name for name in dir(nn) if name not in known and not name.startswith('_') and inspect.isclass(getattr(nn, name)) and issubclass(getattr(nn, name), nn.Module) and getattr(nn, name) is not nn.Module]
    for group, names in groups.items():
        for name in names:
            if name in OPS:
                entries.append(dict(name=name, group=group, defaults=OPS[name], required=[]))
                continue
            cls = getattr(nn, name, None)
            if cls is None:
                continue
            try:
                params = inspect.signature(cls).parameters
            except (TypeError, ValueError):
                continue
            defaults, required = {}, []
            for key, param in params.items():
                if key in ('self', 'device', 'dtype') or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                    continue
                if param.default is inspect.Parameter.empty:
                    required.append(key)
                    defaults[key] = DEFAULTS.get(key)
                else:
                    value = param.default
                    if value is None or isinstance(value, (str, int, float, bool, tuple, list)):
                        defaults[key] = value
            if name == 'Upsample':
                defaults['scale_factor'] = 2
            entries.append(dict(name=name, group=group, defaults=defaults, required=required))
    return entries


def ordered(graph):
    nodes = graph.get('nodes', [])
    if graph.get('version') != 1 or not isinstance(nodes, list) or not 1 <= len(nodes) <= 200:
        raise ValueError('Expected a version 1 graph with 1–200 nodes')
    ids = [n['id'] for n in nodes]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', i) for i in ids):
        raise ValueError('Node identifiers must be unique Python identifiers')
    if not any(n['type'] == 'Input' for n in nodes):
        raise ValueError('At least one input is required')
    pending, result, done = list(nodes), [], set()
    while pending:
        ready = [n for n in pending if all(i in done for i in n.get('inputs', []))]
        if not ready:
            raise ValueError('Graph contains a cycle or missing connection')
        for n in ready:
            if len(n.get('inputs', [])) != len(set(n.get('inputs', []))):
                raise ValueError('Duplicate connections are not supported')
            result.append(n)
            done.add(n['id'])
            pending.remove(n)
    return result


def build(graph):
    if len(json.dumps(graph, allow_nan=False)) > 500000:
        raise ValueError('Graph document is too large')
    nodes = ordered(graph)
    allowed = {c['name'] for c in catalog()}
    class VisualNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict()
            for n in nodes:
                kind = n['type']
                if kind not in allowed | {'Input'}:
                    raise ValueError(f'Unsupported operation: {kind}')
                if kind not in OPS and kind != 'Input':
                    self.layers[n['id']] = getattr(nn, kind)(**n.get('params', {}))
        def forward(self, *args):
            values, report = {}, {}
            iterator = iter(args)
            for n in nodes:
                key, kind, p = n['id'], n['type'], n.get('params', {})
                try:
                    xs = [values[i] for i in n.get('inputs', [])]
                    if kind == 'Input': value = next(iterator)
                    elif not xs: raise ValueError('Connect an input')
                    elif kind == 'Output': value = xs[0]
                    elif kind == 'Add':
                        if len(xs) < 2: raise ValueError('Add needs at least two inputs')
                        value = xs[0]
                        for x in xs[1:]: value = value + x
                    elif kind == 'Multiply':
                        if len(xs) < 2: raise ValueError('Multiply needs at least two inputs')
                        value = xs[0]
                        for x in xs[1:]: value = value * x
                    elif kind == 'Concat': value = torch.cat(xs, dim=p.get('dim', 1))
                    elif kind == 'Reshape': value = xs[0].reshape(p['shape'])
                    elif kind == 'Permute': value = xs[0].permute(*p['dims'])
                    else:
                        # Dimensions derived from connected tensors are authoritative.
                        derived = {}
                        if kind in ('Linear', 'Bilinear'):
                            derived['in_features' if kind == 'Linear' else 'in1_features'] = xs[0].shape[-1]
                            if kind == 'Bilinear' and len(xs) > 1: derived['in2_features'] = xs[1].shape[-1]
                        elif kind.startswith(('Conv1', 'Conv2', 'Conv3', 'ConvTranspose')):
                            derived['in_channels'] = xs[0].shape[1] if xs[0].ndim == int(kind[-2])+2 else xs[0].shape[0]
                        elif kind.startswith(('BatchNorm', 'InstanceNorm')): derived['num_features'] = xs[0].shape[1]
                        elif kind == 'GroupNorm': derived['num_channels'] = xs[0].shape[1]
                        if any(p.get(k) != v for k, v in derived.items()):
                            p.update(derived)
                            n['params'] = p
                            self.layers[key] = getattr(nn, kind)(**p)
                            self.layers[key].eval()
                        value = self.layers[key](*xs)
                    values[key] = value
                    tensors = [value] if torch.is_tensor(value) else [v for v in value if torch.is_tensor(v)]
                    report[key] = {'shape': list(tensors[0].shape) if tensors else [], 'dtype': str(tensors[0].dtype) if tensors else '', 'bytes': sum(t.numel() * t.element_size() for t in tensors)}
                except Exception as exc:
                    raise ValueError(f'{key} ({kind}): {exc}') from exc
            return values, report
    return VisualNetwork(), nodes


def analyze(graph):
    with torch.device('meta'):
        net, nodes = build(graph)
        inputs = []
        for n in nodes:
            if n['type'] != 'Input': continue
            p = n.get('params', {})
            shape = p.get('shape', [1, 4])
            if not isinstance(shape, list) or not 1 <= len(shape) <= 8 or any(type(d) is not int or d < 1 or d > 100000 for d in shape):
                raise ValueError('Input shape requires 1–8 positive dimensions, each at most 100000')
            dtype = p.get('dtype', 'float32')
            if dtype not in ('float32', 'float64', 'float16', 'bfloat16', 'int64', 'int32', 'bool'):
                raise ValueError('Unsupported dtype')
            inputs.append(torch.empty(shape, dtype=getattr(torch, dtype), device='meta'))
        net.eval()
        _, report = net(*inputs)
        parameters = sum(p.numel() for p in net.parameters())
        weights = sum(p.numel() * p.element_size() for p in net.parameters())
        buffers = sum(p.numel() * p.element_size() for p in net.buffers())
        activations = sum(v['bytes'] for v in report.values())
    return dict(nodes=report, parameters=parameters, weight_bytes=weights, buffer_bytes=buffers, activation_bytes=activations, inference_bytes=weights+buffers+activations, training_bytes=weights*4+buffers+activations*2, inferred={n['id']: n.get('params', {}) for n in nodes})


def source(graph, name):
    nodes = ordered(graph)
    lines = ['from torch import nn', 'import torch', 'from backend.models.base import ModelBase', '', f'class {name}(ModelBase):', f'    model_id = {name!r}', f'    display_name = {graph.get("name", name)!r}', '    description = "Created with Network Builder"', '    def __init__(self):', '        super().__init__()', '        self.layers = nn.ModuleDict()']
    for n in nodes:
        if n['type'] not in OPS and n['type'] != 'Input':
            lines.append(f"        self.layers[{n['id']!r}] = nn.{n['type']}(**{n.get('params', {})!r})")
    inputs = [n['id'] for n in nodes if n['type'] == 'Input']
    lines += ['', f'    def forward(self, {", ".join(inputs)}):', '        v = {}']
    for n in nodes:
        key, kind, p = n['id'], n['type'], n.get('params', {})
        xs = [f'v[{i!r}]' for i in n.get('inputs', [])]
        if kind == 'Input': expr = key
        elif kind == 'Output': expr = xs[0]
        elif kind in ('Add', 'Multiply'): expr = (' + ' if kind == 'Add' else ' * ').join(xs)
        elif kind == 'Concat': expr = f'torch.cat([{", ".join(xs)}], dim={p.get("dim", 1)!r})'
        elif kind == 'Reshape': expr = f'{xs[0]}.reshape({p["shape"]!r})'
        elif kind == 'Permute': expr = f'{xs[0]}.permute(*{p["dims"]!r})'
        else: expr = f'self.layers[{key!r}]({", ".join(xs)})'
        lines.append(f'        v[{key!r}] = {expr}')
    outputs = [n['id'] for n in nodes if n['type'] == 'Output']
    if not outputs: raise ValueError('Add and connect an Output node before exporting')
    lines.append('        return ' + ', '.join(f'v[{key!r}]' for key in outputs))
    return '\n'.join(lines) + '\n'


def import_sequential(path, class_name=None):
    tree = ast.parse(path.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not class_name:
        return {'classes': [n.name for n in classes]}
    cls = next((c for c in classes if c.name == class_name), None)
    if cls is None: raise ValueError('Class not found')
    init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
    forward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'), None)
    if not init or not forward: raise ValueError('This class has no editable module constructor and forward method')
    # Only accept an exact sequential wrapper; never silently discard custom forward logic.
    if len(forward.body) != 1 or not isinstance(forward.body[0], ast.Return):
        raise ValueError('Custom forward logic requires a tracing adapter; source was preserved')
    ret = forward.body[0].value
    if not isinstance(ret, ast.Call) or not isinstance(ret.func, ast.Attribute) or not isinstance(ret.func.value, ast.Name) or ret.func.value.id != 'self' or len(ret.args) != 1 or not isinstance(ret.args[0], ast.Name) or ret.args[0].id not in [a.arg for a in forward.args.args[1:]]:
        raise ValueError('Only direct sequential forward wrappers can currently be converted losslessly')
    seq = None
    for n in init.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self' and t.attr == ret.func.attr for t in n.targets):
            seq = n.value
    if not isinstance(seq, ast.Call) or not isinstance(seq.func, ast.Attribute) or seq.func.attr != 'Sequential':
        raise ValueError('Custom model import is not yet supported; source was preserved')
    env = {}
    defaults = init.args.defaults
    for arg, default in zip(init.args.args[-len(defaults):], defaults):
        try: env[arg.arg] = ast.literal_eval(default)
        except (ValueError, TypeError): pass
    def literal(node):
        if isinstance(node, ast.Name) and node.id in env: return env[node.id]
        return ast.literal_eval(node)
    nodes = [dict(id='input', type='Input', params={'shape': [1, 4], 'dtype': 'float32'}, inputs=[], x=60, y=140)]
    for call in seq.args:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute): raise ValueError('Unsupported sequential component')
        kind = call.func.attr
        if kind not in {c['name'] for c in catalog()}: raise ValueError(f'Unsupported module {kind}')
        keys = list(inspect.signature(getattr(nn, kind)).parameters)
        params = {keys[i]: literal(v) for i, v in enumerate(call.args)}
        params.update({k.arg: literal(k.value) for k in call.keywords})
        if len(nodes) == 1 and kind == 'Linear': nodes[0]['params']['shape'] = [1, params['in_features']]
        nodes.append(dict(id=f'n{len(nodes)}', type=kind, params=params, inputs=[nodes[-1]['id']], x=60+240*len(nodes), y=140))
    nodes.append(dict(id='output', type='Output', params={}, inputs=[nodes[-1]['id']], x=60+240*len(nodes), y=140))
    return {'graph': dict(version=1, name=class_name, nodes=nodes)}
