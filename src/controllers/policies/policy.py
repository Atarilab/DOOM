import abc

import numpy as np
import onnxruntime as ort
import torch

# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit


# ======================
# Base Policy Interface
# ======================
class PolicyBase(abc.ABC):
    @abc.abstractmethod
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Run inference and return action as torch.Tensor"""
        pass


# ======================
# TorchScript Wrapper
# ======================
class TorchScriptPolicy(PolicyBase):
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.policy = torch.jit.load(model_path).to(device)
        self.policy.eval()
        self._flatten_rnn_parameters()

    def _flatten_rnn_parameters(self):
        for module in self.policy.modules():
            if isinstance(module, torch.nn.RNNBase):
                module.flatten_parameters()

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy(obs.to(self.device))


# ======================
# ONNX Runtime Wrapper
# ======================
class ONNXPolicy(PolicyBase):
    def __init__(self, model_path: str, device: str = "cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        inp = obs.detach().cpu().numpy().astype(np.float32)
        ort_inputs = {self.input_name: inp}
        ort_outs = self.session.run([self.output_name], ort_inputs)
        return torch.from_numpy(ort_outs[0]).to(self.device)


# ==========================
# TensorRT Runtime Wrapper
# ==========================

# class TRTPolicy:
#     def __init__(self, engine_path):
#         logger = trt.Logger(trt.Logger.WARNING)
#         with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
#             self.engine = runtime.deserialize_cuda_engine(f.read())
#         self.context = self.engine.create_execution_context()

#         # Allocate buffers
#         self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

#     def allocate_buffers(self):
#         inputs, outputs, bindings = [], [], []
#         stream = cuda.Stream()
#         for binding in self.engine:
#             size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
#             dtype = trt.nptype(self.engine.get_binding_dtype(binding))
#             host_mem = cuda.pagelocked_empty(size, dtype)
#             device_mem = cuda.mem_alloc(host_mem.nbytes)
#             bindings.append(int(device_mem))
#             if self.engine.binding_is_input(binding):
#                 inputs.append((host_mem, device_mem))
#             else:
#                 outputs.append((host_mem, device_mem))
#         return inputs, outputs, bindings, stream

#     def __call__(self, obs: torch.Tensor):
#         inp_host, inp_dev = self.inputs[0]
#         out_host, out_dev = self.outputs[0]

#         # Copy input
#         np.copyto(inp_host, obs.detach().cpu().numpy().ravel())

#         # Transfer to GPU, run, fetch output
#         cuda.memcpy_htod_async(inp_dev, inp_host, self.stream)
#         self.context.execute_async_v2(self.bindings, self.stream.handle, None)
#         cuda.memcpy_dtoh_async(out_host, out_dev, self.stream)
#         self.stream.synchronize()

#         # Back to torch
#         return torch.from_numpy(out_host.reshape(1, -1)).to(obs.device)
