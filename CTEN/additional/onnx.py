from training import train
import torch

onnx_model = train.codon_optimization_transformer
onnx_input = torch.randn(1, 1, 32, 32)
onnx_program = torch.onnx.dynamo_export(onnx_model, onnx_input)