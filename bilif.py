import torch
from spikingjelly.activation_based import surrogate

from spikingjelly.activation_based.neuron import LIFNode
from typing import Callable, Optional

from spikingjelly.activation_based.auto_cuda.neuron_kernel import NeuronFPTTKernel, NeuronBPTTKernel, NeuronATGFBase
from spikingjelly.activation_based.auto_cuda.neuron_kernel import cfunction, base

def neuronal_hard_reset(v_next: str, h: str, spike: str, v_reset: str, dtype: str = 'float'):
    if dtype == 'float':
        codes = cfunction.abs(y=f'{dtype} abs_spike', x=spike, dtype=dtype)
        codes += f'{v_next} = {h} * (1.0f - abs_spike) + {v_reset} * abs_spike;'
        return codes
    elif dtype == 'half2':
        codes = cfunction.abs(y=f'{dtype} abs_spike', x=spike, dtype=dtype)
        codes += f'{v_next} = __hfma2({h}, __hsub2(__float2half2_rn(1.0f), abs_spike), __hmul2(v_reset, abs_spike));'
        return codes
    else:
        raise NotImplementedError(dtype)

def neuronal_soft_reset(v_next: str, h: str, spike: str, v_th: str, dtype: str = 'float'):
    if dtype == 'float':
        return f'{v_next} = {h} - {v_th} * {spike};'
    elif dtype == 'half2':
        return f'{v_next} = __hsub2({h}, __hmul2({v_th}, {spike}));'
    else:
        raise NotImplementedError(dtype)
    
def neuronal_fire(spike: str, v: str, v_th: str, dtype: str = 'float'):
    if dtype == 'float':
        codes = cfunction.heaviside(y=f'{dtype} pos_spike', x=f'({v} - {v_th})', dtype=dtype)
        codes += cfunction.heaviside(y=f'{dtype} neg_spike', x=f'(-{v} - {v_th})', dtype=dtype)
        codes += cfunction.sub(z=spike, x='pos_spike', y='neg_spike', dtype=dtype)
        return codes
    elif dtype == 'half2':
        codes = cfunction.heaviside(y=f'{dtype} pos_spike', x=f'__hsub2({v} - {v_th})', dtype=dtype)
        codes += cfunction.heaviside(y=f'{dtype} neg_spike', x=f'__hsub2(-{v} - {v_th})', dtype=dtype)
        codes += cfunction.sub(z=spike, x='pos_spike', y='neg_spike', dtype=dtype)
        return codes
    else:
        raise NotImplementedError(dtype)
    
class TwoWayLIFNodeFPTTKernel(NeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} &', cname='decay')

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(self.neuronal_charge())

        core_codes.append(neuronal_fire(spike='spike_seq[t]', v='h_seq[t]', v_th='v_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(
                neuronal_hard_reset(v_next='v_v_seq[t + dt]', h='h_seq[t]', spike='spike_seq[t]', v_reset='v_reset',
                                    dtype=self.dtype))
        else:
            core_codes.append(
                neuronal_soft_reset(v_next='v_v_seq[t + dt]', h='h_seq[t]', spike='spike_seq[t]', v_th='v_th',
                                    dtype=self.dtype))

        self._core = core_codes.codes
        return self._core
    
    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(z=f'{self.dtype} TwoWayLIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        else:
            codes = f'{self.dtype} TwoWayLIFNodeFPTTKernel_temp_var = v_v_seq[t];'

        if self.decay_input:
            codes += cfunction.sub(z='TwoWayLIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='TwoWayLIFNodeFPTTKernel_temp_var', dtype=self.dtype)
            codes += cfunction.mul(z='TwoWayLIFNodeFPTTKernel_temp_var', x='decay', y='TwoWayLIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='TwoWayLIFNodeFPTTKernel_temp_var', x='decay', y='TwoWayLIFNodeFPTTKernel_temp_var',
                                    dtype=self.dtype)
            codes += cfunction.sub(z='TwoWayLIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='TwoWayLIFNodeFPTTKernel_temp_var',
                                    dtype=self.dtype)

        codes += cfunction.add(z='h_seq[t]', x='TwoWayLIFNodeFPTTKernel_temp_var', y='v_v_seq[t]', dtype=self.dtype)

        return codes
    
class TwoWayLIFNodeBPTTKernel(NeuronBPTTKernel):
    def __init__(self, decay_input: bool, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} &', cname='decay')

    def grad_h_next_to_v(self) -> str:
        return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_v', x=cfunction.constant(None, x=1., dtype=self.dtype), y='decay', dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        if not self.decay_input:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
        else:
            return f'const {self.dtype} grad_h_to_x = decay;'
        
    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(cfunction.sub(z=f'const {self.dtype} pos_over_th', x='h_seq[t]', y='v_th', dtype=self.dtype))
        core_codes.append(cfunction.sub(z=f'const {self.dtype} neg_over_th', x='-h_seq[t]', y='v_th', dtype=self.dtype))
        core_codes.append(cfunction.heaviside(y=f'const {self.dtype} pos_spike_seq_t', x='pos_over_th', dtype=self.dtype))
        core_codes.append(cfunction.heaviside(y=f'const {self.dtype} neg_spike_seq_t', x='neg_over_th', dtype=self.dtype))
        core_codes.append(cfunction.sub(z=f'const {self.dtype} spike_seq_t', x='pos_spike_seq_t', y='neg_spike_seq_t', dtype=self.dtype))
        core_codes.append(self.surrogate_function(y=f'const {self.dtype} grad_pos_s_to_h', x='pos_over_th', dtype=self.dtype))
        core_codes.append(self.surrogate_function(y=f'const {self.dtype} grad_neg_s_to_h', x='neg_over_th', dtype=self.dtype, affix='2'))
        core_codes.append(cfunction.add(z=f'const {self.dtype} grad_s_to_h', x='grad_pos_s_to_h', y='grad_neg_s_to_h', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(cfunction.abs(y=f'{self.dtype} abs_spike', x='spike_seq_t', dtype=self.dtype))
            core_codes.append(
                cfunction.sub(z=f'{self.dtype} grad_v_to_h', x=cfunction.constant(y=None, x=1., dtype=self.dtype),
                                y='abs_spike', dtype=self.dtype))

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.sub(z=f'{self.dtype} temp_var', x='v_reset', y='h_seq[t]', dtype=self.dtype))
                    core_codes.append(cfunction.mul(z=f'temp_var', x='temp_var', y='spike_seq_t', dtype=self.dtype))
                    core_codes.append(cfunction.mul(z=f'temp_var', x='temp_var', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.add(z=f'grad_v_to_h', x='temp_var', y='grad_v_to_h', dtype=self.dtype))


        else:
            core_codes.append(f'{self.dtype} grad_v_to_h = {cfunction.constant(None, 1., dtype=self.dtype)}')

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.mul(z=f'{self.dtype} temp_var', x='v_th', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.sub(z=f'grad_v_to_h', x='grad_v_to_h', y='temp_var', dtype=self.dtype))

        core_codes.append(self.grad_h_next_to_v())
        core_codes.append(cfunction.mul(z=f'grad_h', x='grad_h', y='grad_h_next_to_v', dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_h', x='grad_v_seq[t]', y='grad_h', dtype=self.dtype))
        core_codes.append(cfunction.mul(z='grad_h', x='grad_h', y='grad_v_to_h', dtype=self.dtype))
        with base.CodeBlock(core_codes):
            core_codes.append(
                cfunction.mul(z=f'{self.dtype} temp_var', x='grad_spike_seq[t]', y='grad_s_to_h', dtype=self.dtype))
            core_codes.append(cfunction.add(z='grad_h', x='grad_h', y='temp_var', dtype=self.dtype))

        core_codes.append(self.grad_h_to_x())
        core_codes.append(cfunction.mul(z='grad_x_seq[t]', x='grad_h', y='grad_h_to_x', dtype=self.dtype))

        self._core = core_codes.codes
        return self._core

class TwoWayLIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None, decay: float,
                forward_kernel: TwoWayLIFNodeFPTTKernel, backward_kernel: TwoWayLIFNodeBPTTKernel):
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'decay': decay,
        }
        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                            numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                            backward_kernel=backward_kernel, decay=py_dict['decay'])


        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        py_dict['decay'] = ctx.decay

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')


        backward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None


        return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None, None

class TwoWayLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                    v_reset: Optional[float] = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                    detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

    def neuronal_fire(self):
        pos_fire = self.surrogate_function(self.v - self.v_threshold)
        neg_fire = self.surrogate_function(-self.v - self.v_threshold)
        return pos_fire - neg_fire
    
    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        if spike > 0:
            v = (1. - spike) * v + spike * v_reset
        else:
            v = (1. + spike) * v - spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v
    
    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)
            
    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        pos_spike = (v >= v_threshold).to(x)
        neg_spike = (v <= -v_threshold).to(x)
        v = v_reset * pos_spike + v_reset * neg_spike + (1. - pos_spike - neg_spike) * v
        return pos_spike - neg_spike, v


    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                                v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        pos_spike = (v >= v_threshold).to(x)
        neg_spike = (v <= -v_threshold).to(x)
        v = v_reset * pos_spike + v_reset * neg_spike + (1. - pos_spike - neg_spike) * v
        return pos_spike - neg_spike, v


    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            tau: float):
        v = v + (x - v) / tau
        pos_spike = (v >= v_threshold).to(x)
        neg_spike = (v <= -v_threshold).to(x)
        v = v - (pos_spike - neg_spike) * v_threshold
        return pos_spike - neg_spike, v


    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                                tau: float):
        v = v * (1. - 1. / tau) + x
        pos_spike = (v >= v_threshold).to(x)
        neg_spike = (v <= -v_threshold).to(x)
        v = v - (pos_spike - neg_spike) * v_threshold
        return pos_spike - neg_spike, v


    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            pos_spike = (v >= v_threshold).to(x_seq)
            neg_spike = (v <= -v_threshold).to(x_seq)
            v = v_reset * pos_spike + v_reset * neg_spike + (1. - pos_spike - neg_spike) * v
            spike_seq[t] = pos_spike - neg_spike
        return spike_seq, v


    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                        v_threshold: float, v_reset: float, tau: float):
        
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            pos_spike = (v >= v_threshold).to(x_seq)
            neg_spike = (v <= -v_threshold).to(x_seq)
            v = v_reset * pos_spike + v_reset * neg_spike + (1. - pos_spike - neg_spike) * v
            spike_seq[t] = pos_spike - neg_spike
            v_seq[t] = v
        return spike_seq, v, v_seq


    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                                v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            pos_spike = (v >= v_threshold).to(x_seq)
            neg_spike = (v <= -v_threshold).to(x_seq)
            v = v_reset * pos_spike + v_reset * neg_spike + (1. - pos_spike - neg_spike) * v
            spike_seq[t] = pos_spike - neg_spike
        return spike_seq, v


    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                            v_threshold: float, v_reset: float,
                                                                            tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            pos_spike = (v >= v_threshold).to(x_seq)
            neg_spike = (v <= -v_threshold).to(x_seq)
            v = v_reset * pos_spike + v_reset * neg_spike + (1. - pos_spike - neg_spike) * v
            spike_seq[t] = pos_spike - neg_spike
            v_seq[t] = v
        return spike_seq, v, v_seq


    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            pos_spike = (v >= v_threshold).to(x_seq)
            neg_spike = (v <= -v_threshold).to(x_seq)
            v = v - (pos_spike - neg_spike) * v_threshold
            spike_seq[t] = pos_spike - neg_spike
        return spike_seq, v


    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                        v_threshold: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            pos_spike = (v >= v_threshold).to(x_seq)
            neg_spike = (v <= -v_threshold).to(x_seq)
            v = v - (pos_spike - neg_spike) * v_threshold
            spike_seq[t] = pos_spike - neg_spike
            v_seq[t] = v
        return spike_seq, v, v_seq


    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                                tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            pos_spike = (v >= v_threshold).to(x_seq)
            neg_spike = (v <= -v_threshold).to(x_seq)
            v = v - (pos_spike - neg_spike) * v_threshold
            spike_seq[t] = pos_spike - neg_spike
        return spike_seq, v


    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                            v_threshold: float,
                                                                            tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            pos_spike = (v >= v_threshold).to(x_seq)
            neg_spike = (v <= -v_threshold).to(x_seq)
            v = v - (pos_spike - neg_spike) * v_threshold
            spike_seq[t] = pos_spike - neg_spike
            v_seq[t] = v
        return spike_seq, v, v_seq
    
    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            return spike
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x, self.v,
                                                                                            self.v_threshold, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.tau)
            else:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(x, self.v,
                                                                                            self.v_threshold,
                                                                                            self.v_reset, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.v_reset,
                                                                                                self.tau)
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                T = x_seq.shape[0]
                y_seq = []
                if self.store_v_seq:
                    v_seq = []
                for t in range(T):
                    y = self.single_step_forward(x_seq[t])
                    y_seq.append(y)
                    if self.store_v_seq:
                        v_seq.append(self.v)

                if self.store_v_seq:
                    self.v_seq = torch.stack(v_seq)

                return torch.stack(y_seq)
            else:
                hard_reset = self.v_reset is not None
                if x_seq.dtype == torch.float:
                    dtype = 'float'
                elif x_seq.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x_seq.dtype)

                if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset, dtype=dtype, decay_input=self.decay_input):
                    self.forward_kernel = TwoWayLIFNodeFPTTKernel(decay_input=self.decay_input, hard_reset=hard_reset, dtype=dtype)

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype, decay_input=self.decay_input):
                    self.backward_kernel = TwoWayLIFNodeBPTTKernel(decay_input=self.decay_input, surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset, detach_reset=self.detach_reset, dtype=dtype)

                self.v_float_to_tensor(x_seq[0])

                spike_seq, v_seq = TwoWayLIFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(0),
                                                                    self.v_threshold, self.v_reset, 1. / self.tau,
                                                                    self.forward_kernel,
                                                                    self.backward_kernel)

                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)

                if self.store_v_seq:
                    self.v_seq = v_seq

                self.v = v_seq[-1].clone()

                return spike_seq
                
        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_decay_input(x_seq, self.v,
                                                                                                    self.v_threshold,
                                                                                                    self.tau)
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq, self.v,
                                                                                                        self.v_threshold,
                                                                                                        self.tau)
            else:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_decay_input(x_seq, self.v,
                                                                                                    self.v_threshold,
                                                                                                    self.v_reset,
                                                                                                    self.tau)
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq, self.v,
                                                                                                        self.v_threshold,
                                                                                                        self.v_reset,
                                                                                                        self.tau)
            return spike_seq