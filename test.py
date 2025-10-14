import torch
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1012969)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(2, 10), stride=(10, 1), dtype=float64, device=cuda
    var_node_4 = arg_1 # size=(10, 3), stride=(3, 1), dtype=float64, device=cuda
    var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_4.to(torch.float64)) # size=(2, 3), stride=(3, 1), dtype=float64, device=cuda
    _inp_unique_wide = torch.arange(1, device=var_node_2.device, dtype=torch.int64)
    _uniq_wide = torch.unique(_inp_unique_wide)
    var_node_1 = _uniq_wide.to(var_node_2.dtype) # size=(1,), stride=(1,), dtype=float64, device=cuda
    var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64) # size=(1, 18), stride=(18, 1), dtype=float64, device=cuda
    var_node_0 = torch.matmul(var_node_1.to(torch.float64), var_node_5.to(torch.float64)) # size=(18,), stride=(1,), dtype=float64, device=cuda
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randn(20).to(torch.float64), (2, 10), (10, 1))
arg_1 = torch.as_strided(torch.randn(30).to(torch.float64), (10, 3), (3, 1))

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print('✅ compile success')
print('✅ results match', torch.allclose(result_original, result_compiled))