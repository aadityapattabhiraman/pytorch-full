# Document Title

#### Parameter 
class torch.nn.parameter.Parameter(data=None, requires_grad=True)

A kind of Tensor that is to be considered a module parameter.
Parameters are Tensor subclasses, that have a very special property when used with Module s - when they’re assigned as Module attributes they are automatically added to the list of its parameters, and will appear e.g. in parameters() iterator. Assigning a Tensor doesn’t have such effect. This is because one might want to cache some temporary state, like last hidden state of the RNN, in the model. If there was no such class as Parameter, these temporaries would get registered too.

Parameters
    data (Tensor) – parameter tensor.
    requires_grad (bool, optional) – if the parameter requires gradient. Note that the torch.no_grad() context does NOT affect the default behavior of Parameter creation–the Parameter will still have requires_grad=True in no_grad mode. See Locally disabling gradient computation for more details. 
    Default: True

#### Module
class torch.nn.Module(*args, **kwargs)

Base class for all neural network modules.
Your models should also subclass this class.
Modules can also contain other Modules, allowing to nest them in a tree structure.

#### forward
forward(*input)

Define the computation performed at every call.
Should be overridden by all subclasses.

#### cuda
cuda(device=None)

Move all model parameters and buffers to the GPU.
This also makes associated parameters and buffers different objects. So it should be called before constructing optimizer if the module will live on GPU while being optimized.

#### sequential
class torch.nn.Sequential(*args: Module)

A sequential container.
Modules will be added to it in the order they are passed in the constructor. Alternatively, an OrderedDict of modules can be passed in. The forward() method of Sequential accepts any input and forwards it to the first module it contains. It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
The value a Sequential provides over manually calling a sequence of modules is that it allows treating the whole container as a single module, such that performing a transformation on the Sequential applies to each of the modules it stores (which are each a registered submodule of the Sequential).
What’s the difference between a Sequential and a torch.nn.ModuleList? A ModuleList is exactly what it sounds like–a list for storing Module s! On the other hand, the layers in a Sequential are connected in a cascading way.

#### ModuleList
class torch.nn.ModuleList(modules=None)

Holds submodules in a list.
ModuleList can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all Module methods.

Parameters
    modules (iterable, optional) – an iterable of modules to add

#### ModuleDict
class torch.nn.ModuleDict(modules=None)

Holds submodules in a dictionary.
ModuleDict can be indexed like a regular Python dictionary, but modules it contains are properly registered, and will be visible by all Module methods.

ModuleDict is an ordered dictionary that respects
    the order of insertion, and
    in update(), the order of the merged OrderedDict, dict (started from Python 3.6) or another ModuleDict (the argument to update()).

Note that update() with other unordered mapping types (e.g., Python’s plain dict before Python version 3.6) does not preserve the order of the merged mapping.

Parameters
    modules (iterable, optional) – a mapping (dictionary) of (string: module) or an iterable of key-value pairs of type (string, module)

#### ParameterList
#### ParameterDict

### Convolution Layers

#### conv1d
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,  padding_mode='zeros', device=None, dtype=None)

Applies a 1D convolution over an input signal composed of several input planes.
stride controls the stride for the cross-correlation, a single number or a one-element tuple.

padding controls the amount of padding applied to the input. It can be either a string {‘valid’, ‘same’} or a tuple of ints giving the amount of implicit padding applied on both sides.

dilation controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this link has a nice visualization of what dilation does.

groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups. For example,
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels and producing half the output channels, and both subsequently concatenated.
        At groups= in_channels, each input channel is convolved with its own set of filters (of size
        
Parameters
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) – Padding added to both sides of the input. Default: 0
        padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

