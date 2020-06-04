import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
from tensorflow_compression.python.layers import parameterizers
from tensorflow_compression.python.ops import padding_ops
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import range_coding_ops
from tensorflow.python.keras.engine import input_spec
import numpy as np

class Intercept(tf.layers.Layer):
  def __init__(self, start, stop, step=1):
    super(Intercept, self).__init__()
    self.start = start
    self.stop = stop
    self.step = step
    self.const = tf.constant([1]*stop+[0]*(stop-start),dtype=tf.float32)

  def build(self, input_shape):
    #tf.set_random_seed(self.seed)
    super(Intercept, self).build(input_shape)

  def call(self, inputs):
    mask = tf.random_crop(self.const, [self.stop])
    rate = tf.reduce_sum(mask) / self.stop
    output = inputs * mask / rate # broadcast (?,16,16,256) * (256,)
    return output


class InterceptNorate(tf.layers.Layer):
  def __init__(self, start, stop, step=1):
    super(InterceptNorate, self).__init__()
    self.start = start
    self.stop = stop
    self.step = step
    self.const = tf.constant([1]*stop+[0]*(stop-start),dtype=tf.float32)

  def build(self, input_shape):
    #tf.set_random_seed(self.seed)
    super(InterceptNorate, self).build(input_shape)

  def call(self, inputs):
    mask = tf.random_crop(self.const, [self.stop])
    rate = tf.reduce_sum(mask) / self.stop
    output = inputs * mask # broadcast (?,16,16,256) * (256,)
    return output


class OneGDN(tfc.GDN):
  def __init__(self, *args, **kwargs):
    super(OneGDN, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    super(OneGDN, self).build(input_shape)

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    ndim = self._input_rank

    if self.rectify:
      inputs = tf.nn.relu(inputs)

    # Compute normalization pool.
    if ndim == 2:
      norm_pool = tf.linalg.matmul(tf.math.abs(inputs), self.gamma)
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    elif self.data_format == "channels_last" and ndim <= 4:
      # TODO(unassigned): This branch should also work for ndim == 5, but
      # currently triggers a bug in TF.
      shape = self.gamma.shape.as_list()
      gamma = tf.reshape(self.gamma, (ndim - 2) * [1] + shape)
      norm_pool = tf.nn.convolution(tf.math.abs(inputs), gamma, "VALID")
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    else:  # generic implementation
      # This puts channels in the last dimension regardless of input.
      norm_pool = tf.linalg.tensordot(
          tf.math.abs(inputs), self.gamma, [[self._channel_axis()], [0]])
      norm_pool += self.beta
      if self.data_format == "channels_first":
        # Return to channels_first format if necessary.
        axes = list(range(ndim - 1))
        axes.insert(1, ndim - 1)
        norm_pool = tf.transpose(norm_pool, axes)

    if self.inverse:
      pass
    else:
      norm_pool = tf.math.reciprocal(norm_pool)
    outputs = inputs * norm_pool

    if not tf.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))
    return outputs


class DynamicCondSignalConv2D(tfc.SignalConv2D):
  def __init__(self, *args, **kwargs):
    super(DynamicCondSignalConv2D, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.fc_u =  tf.keras.layers.Dense(self.filters, activation=tf.nn.softplus, name="fc_u")
    self.fc_v =  tf.keras.layers.Dense(self.filters, activation=None, name="fc_v")
    super(DynamicCondSignalConv2D, self).build(input_shape)
    self.input_spec = None
  
  def call(self, inputs, lmbda, active_out):
    inputs = tf.convert_to_tensor(inputs)
    active_in = tf.shape(inputs)[-1]
    self.active_out = active_out
    outputs = inputs

    # Not for all possible combinations of (`kernel_support`, `corr`,
    # `strides_up`, `strides_down`) TF ops exist. We implement some additional
    # combinations by manipulating the kernels and toggling `corr`.
    kernel = self.kernel
    corr = self.corr

    # If a convolution with no upsampling is desired, we flip the kernels and
    # use cross correlation to implement it, provided the kernels are odd-length
    # in every dimension (with even-length kernels, the boundary handling
    # would have to change).
    if (not corr and
        all(s == 1 for s in self.strides_up) and
        all(s % 2 == 1 for s in self.kernel_support)):
      corr = True
      slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
      kernel = kernel[slices]

    # Similarly, we can implement a cross correlation using convolutions.
    # However, we do this only if upsampling is requested, as we are potentially
    # wasting computation in the boundaries whenever we call the transpose ops.
    elif (corr and
          any(s != 1 for s in self.strides_up) and
          all(s % 2 == 1 for s in self.kernel_support)):
      corr = False
      slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
      kernel = kernel[slices]

    slices = self._rank * (slice(None),) + (slice(0, active_in), slice(0, active_out))
    kernel = kernel[slices]
    
    # Compute amount of necessary padding, and determine whether to use built-in
    # padding or to pre-pad with a separate op.
    if self.padding == "valid":
      padding = prepadding = self._rank * ((0, 0),)
    else:  # same_*
      padding = padding_ops.same_padding_for_kernel(
          self.kernel_support, corr, self.strides_up)
      if (self.padding == "same_zeros" and
          not self.channel_separable and
          1 <= self._rank <= 2 and
          self.use_explicit):
        # Don't pre-pad and use built-in EXPLICIT mode.
        prepadding = self._rank * ((0, 0),)
      else:
        # Pre-pad and then use built-in valid padding mode.
        outputs = tf.pad(
            outputs, self._padded_tuple(padding, (0, 0)), self._pad_mode)
        prepadding = padding
        padding = self._rank * ((0, 0),)

    # Compute the convolution/correlation. Prefer EXPLICIT padding ops where
    # possible, but don't use them to implement VALID padding.
    if (corr and
        all(s == 1 for s in self.strides_up) and
        not self.channel_separable and
        1 <= self._rank <= 2 and
        not all(p[0] == p[1] == 0 for p in padding)):
      outputs = self._correlate_down_explicit(outputs, kernel, padding)
    elif (corr and
          all(s == 1 for s in self.strides_up) and
          all(p[0] == p[1] == 0 for p in padding)):
      outputs = self._correlate_down_valid(outputs, kernel)
    elif (not corr and
          not self.channel_separable and
          1 <= self._rank <= 2 and
          self.use_explicit):
      outputs = self._up_convolve_transpose_explicit(
          outputs, kernel, prepadding)
    elif not corr:
      outputs = self._up_convolve_transpose_valid(
          outputs, kernel, prepadding)
    else:
      self._raise_notimplemented()

    # Now, add bias if requested.
    if self.use_bias:
      bias = self.bias[slice(0, active_out)]
      if self.data_format == "channels_first":
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients.
        if self._rank == 1:
          # tf.nn.bias_add does not accept a 1D input tensor.
          outputs = tf.expand_dims(outputs, 2)
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
          outputs = tf.squeeze(outputs, [2])
        elif self._rank == 2:
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
        elif self._rank >= 3:
          shape = tf.shape(outputs)
          outputs = tf.reshape(
              outputs, tf.concat([shape[:3], [-1]], axis=0))
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
          outputs = tf.reshape(outputs, shape)
      else:
        outputs = tf.nn.bias_add(outputs, bias)

    s = self.fc_u(lmbda)[slice(None), slice(0, active_out)]
    b = self.fc_v(lmbda)[slice(None), slice(0, active_out)]
    outputs = outputs * s + b

    # Finally, pass through activation function if requested.
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint:disable=not-callable

    # Aid shape inference, for some reason shape info is not always available.
    if not tf.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))
    #print(kernel)
    #print(outputs)

    return outputs
  
  
  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank(self._rank + 2)
    batch = input_shape[0]
    if self.data_format == "channels_first":
      spatial = input_shape[2:].dims
      channels = input_shape[1]
    else:
      spatial = input_shape[1:-1].dims
      channels = input_shape[-1]

    for i, s in enumerate(spatial):
      if self.extra_pad_end:
        s *= self.strides_up[i]
      else:
        s = (s - 1) * self.strides_up[i] + 1
      if self.padding == "valid":
        s -= self.kernel_support[i] - 1
      s = (s - 1) // self.strides_down[i] + 1
      spatial[i] = s

    if self.channel_separable:
      channels *= self.active_out
    else:
      channels = self.active_out

    if self.data_format == "channels_first":
      return tf.TensorShape([batch, None] + spatial)
    else:
      return tf.TensorShape([batch] + spatial + [None])


  def _up_convolve_transpose_valid(self, inputs, kernel, prepadding):
    # Computes upsampling followed by convolution, via transpose convolution ops
    # in VALID mode. This is a relatively inefficient implementation of
    # upsampled convolutions, where we need to crop away a lot of the values
    # computed in the boundaries.

    # Transpose convolutions expect the output and input channels in reversed
    # order. We implement this by swapping those dimensions of the kernel.
    # For channel separable convolutions, we can't currently perform anything
    # other than one filter per channel, so the last dimension needs to be of
    # length one. Since this happens to be the format that the op expects it,
    # we can skip the transpose in that case.
    if not self.channel_separable:
      kernel = tf.transpose(
          kernel, list(range(self._rank)) + [self._rank + 1, self._rank])

    # Compute shape of temporary.
    input_shape = tf.shape(inputs)
    temp_shape = [input_shape[0]] + (self._rank + 1) * [None]
    if self.data_format == "channels_last":
      spatial_axes = range(1, self._rank + 1)
      temp_shape[-1] = (
          input_shape[-1] if self.channel_separable else self.active_out)
    else:
      spatial_axes = range(2, self._rank + 2)
      temp_shape[1] = input_shape[1] if self.channel_separable else self.active_out
    if self.extra_pad_end:
      get_length = lambda l, s, k: l * s + (k - 1)
    else:
      get_length = lambda l, s, k: l * s + ((k - 1) - (s - 1))
    for i, a in enumerate(spatial_axes):
      temp_shape[a] = get_length(
          input_shape[a], self.strides_up[i], self.kernel_support[i])

    data_format = self._op_data_format
    strides = self._padded_tuple(self.strides_up, 1)

    # Compute convolution.
    if self._rank == 1 and not self.channel_separable:
      # There's no 1D equivalent to conv2d_backprop_input, so we insert an
      # extra dimension and use the 2D op.
      extradim = {"channels_first": 2, "channels_last": 1}[self.data_format]
      data_format = data_format.replace("W", "HW")
      strides = strides[:extradim] + (strides[extradim],) + strides[extradim:]
      temp_shape = temp_shape[:extradim] + [1] + temp_shape[extradim:]
      kernel = tf.expand_dims(kernel, 0)
      inputs = tf.expand_dims(inputs, extradim)
      outputs = tf.nn.conv2d_backprop_input(
          temp_shape, kernel, inputs,
          strides=strides, padding="VALID", data_format=data_format)
      outputs = tf.squeeze(outputs, [extradim])
    elif self._rank == 1 and self.channel_separable and self.filters == 1:
      # There's no 1D equivalent to depthwise_conv2d_native_backprop_input, so
      # we insert an extra dimension and use the 2D op.
      extradim = {"channels_first": 2, "channels_last": 1}[self.data_format]
      data_format = data_format.replace("W", "HW")
      strides = strides[:extradim] + (strides[extradim],) + strides[extradim:]
      temp_shape = temp_shape[:extradim] + [1] + temp_shape[extradim:]
      kernel = tf.expand_dims(kernel, 0)
      inputs = tf.expand_dims(inputs, extradim)
      outputs = tf.nn.depthwise_conv2d_native_backprop_input(
          temp_shape, kernel, inputs,
          strides=strides, padding="VALID", data_format=data_format)
      outputs = tf.squeeze(outputs, [extradim])
    elif self._rank == 2 and not self.channel_separable:
      outputs = tf.nn.conv2d_backprop_input(
          temp_shape, kernel, inputs,
          strides=strides, padding="VALID", data_format=data_format)
    elif (self._rank == 2 and self.channel_separable and
          self.filters == 1 and self.strides_up[0] == self.strides_up[1]):
      outputs = tf.nn.depthwise_conv2d_native_backprop_input(
          temp_shape, kernel, inputs,
          strides=strides, padding="VALID", data_format=data_format)
    elif self._rank == 3 and not self.channel_separable:
      outputs = tf.nn.conv3d_transpose(
          inputs, kernel, temp_shape,
          strides=strides, padding="VALID", data_format=data_format)
    else:
      self._raise_notimplemented()

    # Perform crop, taking into account any pre-padding that was applied.
    slices = (self._rank + 2) * [slice(None)]
    for i, a in enumerate(spatial_axes):
      if self.padding == "valid":
        # Take `kernel_support - 1` samples away from both sides. This leaves
        # just samples computed without any padding.
        start = stop = self.kernel_support[i] - 1
      else:  # same
        # Take half of kernel sizes plus the pre-padding away from each side.
        start = prepadding[i][0] * self.strides_up[i]
        start += self.kernel_support[i] // 2
        stop = prepadding[i][1] * self.strides_up[i]
        stop += (self.kernel_support[i] - 1) // 2
      step = self.strides_down[i]
      start = start if start > 0 else None
      stop = -stop if stop > 0 else None
      step = step if step > 1 else None
      slices[a] = slice(start, stop, step)
    if not all(s.start is s.stop is s.step is None for s in slices):
      outputs = outputs[tuple(slices)]

    return outputs

  def _up_convolve_transpose_explicit(self, inputs, kernel, prepadding):
    # Computes upsampling followed by convolution, via transpose convolution ops
    # in EXPLICIT mode. This is an efficient implementation of upsampled
    # convolutions, where we only compute values that are necessary.
    do_cast = inputs.dtype.is_integer

    # conv2d_backprop_input expects the output and input channels in reversed
    # order. We implement this by swapping those dimensions of the kernel.
    kernel = tf.transpose(
        kernel, list(range(self._rank)) + [self._rank + 1, self._rank])

    # Compute explicit padding corresponding to the equivalent conv2d call,
    # and the shape of the output, taking into account any pre-padding.
    input_shape = tf.shape(inputs)
    padding = (self._rank + 2) * [(0, 0)]
    output_shape = [input_shape[0]] + (self._rank + 1) * [None]
    if self.data_format == "channels_last":
      spatial_axes = range(1, self._rank + 1)
      output_shape[-1] = self.active_out
    else:
      spatial_axes = range(2, self._rank + 2)
      output_shape[1] = self.active_out
    if self.extra_pad_end:
      get_length = lambda l, s, k, p: l * s + ((k - 1) - p)
    else:
      get_length = lambda l, s, k, p: l * s + ((k - 1) - (s - 1) - p)
    for i, a in enumerate(spatial_axes):
      if self.padding == "valid":
        padding[a] = 2 * (self.kernel_support[i] - 1,)
      else:  # same
        padding[a] = (
            prepadding[i][0] * self.strides_up[i] + self.kernel_support[i] // 2,
            prepadding[i][1] * self.strides_up[i] + (
                self.kernel_support[i] - 1) // 2,
        )
      output_shape[a] = get_length(
          input_shape[a], self.strides_up[i], self.kernel_support[i],
          sum(padding[a]))

    data_format = self._op_data_format
    strides = self._padded_tuple(self.strides_up, 1)

    # Compute convolution.
    if self._rank == 1 and not self.channel_separable:
      # There's no 1D equivalent to conv2d_backprop_input, so we insert an
      # extra dimension and use the 2D op.
      extradim = {"channels_first": 2, "channels_last": 1}[self.data_format]
      data_format = data_format.replace("W", "HW")
      strides = strides[:extradim] + (strides[extradim],) + strides[extradim:]
      padding = padding[:extradim] + [(0, 0)] + padding[extradim:]
      output_shape = output_shape[:extradim] + [1] + output_shape[extradim:]
      kernel = tf.expand_dims(kernel, 0)
      inputs = tf.expand_dims(inputs, extradim)
      if do_cast:
        inputs = tf.cast(inputs, tf.float32)
      outputs = tf.nn.conv2d_backprop_input(
          output_shape, kernel, inputs,
          strides=strides, padding=padding, data_format=data_format)
      if do_cast:
        outputs = tf.cast(tf.math.round(outputs), self.accum_dtype)
      outputs = tf.squeeze(outputs, [extradim])
    elif self._rank == 2 and not self.channel_separable:
      if do_cast:
        inputs = tf.cast(inputs, tf.float32)
      outputs = tf.nn.conv2d_backprop_input(
          output_shape, kernel, inputs,
          strides=strides, padding=padding, data_format=data_format)
      if do_cast:
        outputs = tf.cast(tf.math.round(outputs), self.accum_dtype)
      #print(outputs, kernel, inputs, sep='\n')
      #print()
    else:
      self._raise_notimplemented()

    # Perform downsampling if it is requested.
    if any(s > 1 for s in self.strides_down):
      slices = tuple(slice(None, None, s) for s in self.strides_down)
      slices = self._padded_tuple(slices, slice(None))
      outputs = outputs[slices]

    return outputs



class CondSignalConv2D(tfc.SignalConv2D):
  def __init__(self, *args, **kwargs):
    super(CondSignalConv2D, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.fc_u =  tf.keras.layers.Dense(self.filters, activation=tf.nn.softplus, name="fc_u")
    self.fc_v =  tf.keras.layers.Dense(self.filters, activation=None, name="fc_v")
    super(CondSignalConv2D, self).build(input_shape)
  
  def call(self, inputs, lmbda):
    inputs = tf.convert_to_tensor(inputs)
    outputs = inputs

    # Not for all possible combinations of (`kernel_support`, `corr`,
    # `strides_up`, `strides_down`) TF ops exist. We implement some additional
    # combinations by manipulating the kernels and toggling `corr`.
    kernel = self.kernel
    corr = self.corr

    # If a convolution with no upsampling is desired, we flip the kernels and
    # use cross correlation to implement it, provided the kernels are odd-length
    # in every dimension (with even-length kernels, the boundary handling
    # would have to change).
    if (not corr and
        all(s == 1 for s in self.strides_up) and
        all(s % 2 == 1 for s in self.kernel_support)):
      corr = True
      slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
      kernel = kernel[slices]

    # Similarly, we can implement a cross correlation using convolutions.
    # However, we do this only if upsampling is requested, as we are potentially
    # wasting computation in the boundaries whenever we call the transpose ops.
    elif (corr and
          any(s != 1 for s in self.strides_up) and
          all(s % 2 == 1 for s in self.kernel_support)):
      corr = False
      slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
      kernel = kernel[slices]

    # Compute amount of necessary padding, and determine whether to use built-in
    # padding or to pre-pad with a separate op.
    if self.padding == "valid":
      padding = prepadding = self._rank * ((0, 0),)
    else:  # same_*
      padding = padding_ops.same_padding_for_kernel(
          self.kernel_support, corr, self.strides_up)
      if (self.padding == "same_zeros" and
          not self.channel_separable and
          1 <= self._rank <= 2 and
          self.use_explicit):
        # Don't pre-pad and use built-in EXPLICIT mode.
        prepadding = self._rank * ((0, 0),)
      else:
        # Pre-pad and then use built-in valid padding mode.
        outputs = tf.pad(
            outputs, self._padded_tuple(padding, (0, 0)), self._pad_mode)
        prepadding = padding
        padding = self._rank * ((0, 0),)

    # Compute the convolution/correlation. Prefer EXPLICIT padding ops where
    # possible, but don't use them to implement VALID padding.
    if (corr and
        all(s == 1 for s in self.strides_up) and
        not self.channel_separable and
        1 <= self._rank <= 2 and
        not all(p[0] == p[1] == 0 for p in padding)):
      outputs = self._correlate_down_explicit(outputs, kernel, padding)
    elif (corr and
          all(s == 1 for s in self.strides_up) and
          all(p[0] == p[1] == 0 for p in padding)):
      outputs = self._correlate_down_valid(outputs, kernel)
    elif (not corr and
          not self.channel_separable and
          1 <= self._rank <= 2 and
          self.use_explicit):
      outputs = self._up_convolve_transpose_explicit(
          outputs, kernel, prepadding)
    elif not corr:
      outputs = self._up_convolve_transpose_valid(
          outputs, kernel, prepadding)
    else:
      self._raise_notimplemented()

    # Now, add bias if requested.
    if self.use_bias:
      bias = self.bias
      if self.data_format == "channels_first":
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients.
        if self._rank == 1:
          # tf.nn.bias_add does not accept a 1D input tensor.
          outputs = tf.expand_dims(outputs, 2)
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
          outputs = tf.squeeze(outputs, [2])
        elif self._rank == 2:
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
        elif self._rank >= 3:
          shape = tf.shape(outputs)
          outputs = tf.reshape(
              outputs, tf.concat([shape[:3], [-1]], axis=0))
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
          outputs = tf.reshape(outputs, shape)
      else:
        outputs = tf.nn.bias_add(outputs, bias)

    s = self.fc_u(lmbda)
    b = self.fc_v(lmbda)
    outputs = outputs * s + b

    # Finally, pass through activation function if requested.
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint:disable=not-callable

    # Aid shape inference, for some reason shape info is not always available.
    if not tf.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))

    return outputs



class Cond1SignalConv2D(tfc.SignalConv2D):
  def __init__(self, *args, **kwargs):
    super(Cond1SignalConv2D, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.fc1 =  tf.keras.layers.Dense(16, activation=tf.nn.relu)
    self.fc2 =  tf.keras.layers.Dense(self.filters, activation=tf.nn.softplus)
    self.fc3 =  tf.keras.layers.Dense(self.filters, activation=None)
    super(Cond1SignalConv2D, self).build(input_shape)
  
  def call(self, inputs, lmbda):
    inputs = tf.convert_to_tensor(inputs)
    outputs = inputs

    # Not for all possible combinations of (`kernel_support`, `corr`,
    # `strides_up`, `strides_down`) TF ops exist. We implement some additional
    # combinations by manipulating the kernels and toggling `corr`.
    kernel = self.kernel
    corr = self.corr

    # If a convolution with no upsampling is desired, we flip the kernels and
    # use cross correlation to implement it, provided the kernels are odd-length
    # in every dimension (with even-length kernels, the boundary handling
    # would have to change).
    if (not corr and
        all(s == 1 for s in self.strides_up) and
        all(s % 2 == 1 for s in self.kernel_support)):
      corr = True
      slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
      kernel = kernel[slices]

    # Similarly, we can implement a cross correlation using convolutions.
    # However, we do this only if upsampling is requested, as we are potentially
    # wasting computation in the boundaries whenever we call the transpose ops.
    elif (corr and
          any(s != 1 for s in self.strides_up) and
          all(s % 2 == 1 for s in self.kernel_support)):
      corr = False
      slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
      kernel = kernel[slices]

    # Compute amount of necessary padding, and determine whether to use built-in
    # padding or to pre-pad with a separate op.
    if self.padding == "valid":
      padding = prepadding = self._rank * ((0, 0),)
    else:  # same_*
      padding = padding_ops.same_padding_for_kernel(
          self.kernel_support, corr, self.strides_up)
      if (self.padding == "same_zeros" and
          not self.channel_separable and
          1 <= self._rank <= 2 and
          self.use_explicit):
        # Don't pre-pad and use built-in EXPLICIT mode.
        prepadding = self._rank * ((0, 0),)
      else:
        # Pre-pad and then use built-in valid padding mode.
        outputs = tf.pad(
            outputs, self._padded_tuple(padding, (0, 0)), self._pad_mode)
        prepadding = padding
        padding = self._rank * ((0, 0),)

    # Compute the convolution/correlation. Prefer EXPLICIT padding ops where
    # possible, but don't use them to implement VALID padding.
    if (corr and
        all(s == 1 for s in self.strides_up) and
        not self.channel_separable and
        1 <= self._rank <= 2 and
        not all(p[0] == p[1] == 0 for p in padding)):
      outputs = self._correlate_down_explicit(outputs, kernel, padding)
    elif (corr and
          all(s == 1 for s in self.strides_up) and
          all(p[0] == p[1] == 0 for p in padding)):
      outputs = self._correlate_down_valid(outputs, kernel)
    elif (not corr and
          not self.channel_separable and
          1 <= self._rank <= 2 and
          self.use_explicit):
      outputs = self._up_convolve_transpose_explicit(
          outputs, kernel, prepadding)
    elif not corr:
      outputs = self._up_convolve_transpose_valid(
          outputs, kernel, prepadding)
    else:
      self._raise_notimplemented()

    # Now, add bias if requested.
    if self.use_bias:
      bias = self.bias
      if self.data_format == "channels_first":
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients.
        if self._rank == 1:
          # tf.nn.bias_add does not accept a 1D input tensor.
          outputs = tf.expand_dims(outputs, 2)
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
          outputs = tf.squeeze(outputs, [2])
        elif self._rank == 2:
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
        elif self._rank >= 3:
          shape = tf.shape(outputs)
          outputs = tf.reshape(
              outputs, tf.concat([shape[:3], [-1]], axis=0))
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
          outputs = tf.reshape(outputs, shape)
      else:
        outputs = tf.nn.bias_add(outputs, bias)

    oh = self.fc1(tf.reshape(lmbda, [1,1]))
    s = self.fc2(oh)
    b = self.fc3(oh)
    outputs = outputs * s + b

    # Finally, pass through activation function if requested.
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint:disable=not-callable

    # Aid shape inference, for some reason shape info is not always available.
    if not tf.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))

    return outputs



class Cond0SignalConv2D(tfc.SignalConv2D):
  def __init__(self, *args, **kwargs):
    super(Cond0SignalConv2D, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.fc1 =  tf.keras.layers.Dense(16, activation=tf.nn.relu)
    self.fc2 =  tf.keras.layers.Dense(self.filters, activation=tf.nn.softplus)
    self.fc3 =  tf.keras.layers.Dense(self.filters, activation=None)
    super(Cond0SignalConv2D, self).build(input_shape)

  def call(self, inputs, lmbda):
    wx = super(Cond0SignalConv2D, self).call(inputs)
    oh = self.fc1(tf.reshape(lmbda, [1,1]))
    s = self.fc2(oh)
    b = self.fc3(oh)
    return wx * s + b


class DynamicSignalConv2D(tfc.SignalConv2D):
  def __init__(self, *args, **kwargs):
    super(DynamicSignalConv2D, self).__init__(*args, **kwargs)
    self.active_in_filters = None
    self.active_out_filters = self.filters

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    super(DynamicSignalConv2D, self).build(input_shape)
    self.input_spec = None


  def call(self, inputs):

    inputs = tf.convert_to_tensor(inputs)
    outputs = inputs

    input_shape = inputs.shape
    channel_axis = {"channels_first": 1, "channels_last": -1}[self.data_format]
    self.active_in_filters = input_shape.as_list()[channel_axis]
    # effects 4 up/down conv methods, and compute_output_shape
    self._filters = self.active_out_filters

    # Not for all possible combinations of (`kernel_support`, `corr`,
    # `strides_up`, `strides_down`) TF ops exist. We implement some additional
    # combinations by manipulating the kernels and toggling `corr`.
    kernel = self.kernel
    corr = self.corr

    # If a convolution with no upsampling is desired, we flip the kernels and
    # use cross correlation to implement it, provided the kernels are odd-length
    # in every dimension (with even-length kernels, the boundary handling
    # would have to change).
    if (not corr and
        all(s == 1 for s in self.strides_up) and
        all(s % 2 == 1 for s in self.kernel_support)):
      corr = True
      slices = self._rank * (slice(None, None, -1),) + (slice(0, self.active_in_filters), slice(0, self.active_out_filters))
      kernel = kernel[slices]

    # Similarly, we can implement a cross correlation using convolutions.
    # However, we do this only if upsampling is requested, as we are potentially
    # wasting computation in the boundaries whenever we call the transpose ops.
    elif (corr and
          any(s != 1 for s in self.strides_up) and
          all(s % 2 == 1 for s in self.kernel_support)):
      corr = False
      slices = self._rank * (slice(None, None, -1),) + (slice(0, self.active_in_filters), slice(0, self.active_out_filters))
      kernel = kernel[slices]
    else:
      slices = self._rank * (slice(None),) + (slice(0, self.active_in_filters), slice(0, self.active_out_filters))
      kernel = kernel[slices]
    

    # Compute amount of necessary padding, and determine whether to use built-in
    # padding or to pre-pad with a separate op.
    if self.padding == "valid":
      padding = prepadding = self._rank * ((0, 0),)
    else:  # same_*
      padding = padding_ops.same_padding_for_kernel(
          self.kernel_support, corr, self.strides_up)
      if (self.padding == "same_zeros" and
          not self.channel_separable and
          1 <= self._rank <= 2 and
          self.use_explicit):
        # Don't pre-pad and use built-in EXPLICIT mode.
        prepadding = self._rank * ((0, 0),)
      else:
        # Pre-pad and then use built-in valid padding mode.
        outputs = tf.pad(
            outputs, self._padded_tuple(padding, (0, 0)), self._pad_mode)
        prepadding = padding
        padding = self._rank * ((0, 0),)
    # Compute the convolution/correlation. Prefer EXPLICIT padding ops where
    # possible, but don't use them to implement VALID padding.
    if (corr and
        all(s == 1 for s in self.strides_up) and
        not self.channel_separable and
        1 <= self._rank <= 2 and
        not all(p[0] == p[1] == 0 for p in padding)):
      outputs = self._correlate_down_explicit(outputs, kernel, padding)
    elif (corr and
          all(s == 1 for s in self.strides_up) and
          all(p[0] == p[1] == 0 for p in padding)):
      outputs = self._correlate_down_valid(outputs, kernel)
    elif (not corr and
          not self.channel_separable and
          1 <= self._rank <= 2 and
          self.use_explicit):
      outputs = self._up_convolve_transpose_explicit(
          outputs, kernel, prepadding)
    elif not corr:
      outputs = self._up_convolve_transpose_valid(
          outputs, kernel, prepadding)
    else:
      self._raise_notimplemented()
    # Now, add bias if requested.
    if self.use_bias:
      bias = self.bias[slice(0,self.active_out_filters)]
      if self.data_format == "channels_first":
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients.
        if self._rank == 1:
          # tf.nn.bias_add does not accept a 1D input tensor.
          outputs = tf.expand_dims(outputs, 2)
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
          outputs = tf.squeeze(outputs, [2])
        elif self._rank == 2:
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
        elif self._rank >= 3:
          shape = tf.shape(outputs)
          outputs = tf.reshape(
              outputs, tf.concat([shape[:3], [-1]], axis=0))
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
          outputs = tf.reshape(outputs, shape)
      else:
        outputs = tf.nn.bias_add(outputs, bias)

    # Finally, pass through activation function if requested.
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint:disable=not-callable

    # Aid shape inference, for some reason shape info is not always available.
    if not tf.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))

    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank(self._rank + 2)
    batch = input_shape[0]
    if self.data_format == "channels_first":
      spatial = input_shape[2:].dims
      channels = input_shape[1]
    else:
      spatial = input_shape[1:-1].dims
      channels = input_shape[-1]

    for i, s in enumerate(spatial):
      if self.extra_pad_end:
        s *= self.strides_up[i]
      else:
        s = (s - 1) * self.strides_up[i] + 1
      if self.padding == "valid":
        s -= self.kernel_support[i] - 1
      s = (s - 1) // self.strides_down[i] + 1
      spatial[i] = s

    if self.channel_separable:
      channels *= self.active_out_filters
    else:
      channels = self.active_out_filters

    if self.data_format == "channels_first":
      return tf.TensorShape([batch, channels] + spatial)
    else:
      return tf.TensorShape([batch] + spatial + [channels])


  def sort_filter(self, sess, vst, sort_in=True, sort_out=True):
    # sort_in: idx/False
    # sort_out: idx/True/False
    # sort in_channel by input idx from input layer
    weights = self.weights
    update_ops = []

    var_kernel = vst[weights[0].name]
    kernel = sess.run(var_kernel).reshape(self.kernel.shape)  # to array, in case of rdft

    if sort_in is not False:
      kernel = kernel[:,:,sort_in,:] # axis=2

    # sort out_channel by calulate L1 norm
    sorted_idx = None
    if sort_out is not False:
      if sort_out is True:
        importance = np.sum(np.abs(kernel), axis=(0,1,2))
        importance[self.active_out_filters:] = np.arange(0, self.active_out_filters-kernel.shape[3], -1)
        sorted_idx = np.argsort(-importance) # descending
      else:
        sorted_idx = sort_out
      kernel = kernel[:,:,:,sorted_idx] # axis=3
      
      if self.use_bias:
        var_bias = vst[weights[1].name] # variable
        op_bias = tf.assign(var_bias, tf.gather(var_bias, sorted_idx, axis=0))
        update_ops.append(op_bias)
      if isinstance(self.activation, DynamicGDN):
        var_beta = vst[weights[2].name]
        var_gamma = vst[weights[3].name]
        op_beta = tf.assign(var_beta, tf.gather(var_beta, sorted_idx, axis=0))
        op_gamma = tf.assign(var_gamma, tf.gather(tf.gather(var_gamma, sorted_idx, axis=0), sorted_idx, axis=1))
        update_ops.extend([op_beta, op_gamma])

    op_kernel = tf.assign(var_kernel, kernel.reshape(weights[0].shape))
    update_ops.append(op_kernel)
    sess.run(update_ops)
    print(sorted_idx)
    return sorted_idx


  def sort_filter_graph(self, sort_in=True, sort_out=True):
    # set weights version of sort_filter, no graph
    # sort_in: idx/False
    # sort_out: idx/True/False
    # sort in_channel by input idx from input layer
    if sort_in is not False:
      self._kernel = tf.gather(self._kernel, sort_in, axis=2)
    # sort out_channel by calulate L1 norm
    sorted_idx = None
    if sort_out is not False:
      if sort_out is True:
        importance = tf.reduce_sum(tf.abs(self.kernel), axis=[0,1,2])
        endpoint = self.kernel.shape.as_list()[3]-self.active_out_filters
        if endpoint > 0:
          protect = tf.range(0, -endpoint, -1.0)
          importance = tf.concat([importance[:self.active_out_filters], protect], axis=0)
        sorted_idx = tf.argsort(importance, direction='DESCENDING')
      else:
        sorted_idx = sort_out 
      self._kernel = tf.gather(self._kernel, sorted_idx, axis=3)
      if self.use_bias:
        self._bias = tf.gather(self._bias, sorted_idx, axis=0)
      if isinstance(self.activation, DynamicGDN):
        self.activation.sort_weight_graph(sorted_idx)
    return sorted_idx



class DynamicEntropyBottleneck(tfc.EntropyBottleneck):
  def __init__(self, *args, **kwargs):
    super(DynamicEntropyBottleneck, self).__init__(*args, **kwargs)
    self.active_out_filters = None

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    super(DynamicEntropyBottleneck, self).build(input_shape)

  def _logits_cumulative(self, inputs, stop_gradient):
    logits = inputs
    _, _, channels, _ = self._get_input_dims()
    self.active_out_filters = channels
    slices = (slice(0, self.active_out_filters), slice(None), slice(None))

    for i in range(len(self.filters) + 1):
      matrix = self._matrices[i][slices]

      if stop_gradient:
        matrix = tf.stop_gradient(matrix)
      logits = tf.linalg.matmul(matrix, logits)

      bias = self._biases[i][slices]
      if stop_gradient:
        bias = tf.stop_gradient(bias)
      logits += bias

      if i < len(self._factors):
        factor = self._factors[i][slices]
        if stop_gradient:
          factor = tf.stop_gradient(factor)
        logits += factor * tf.math.tanh(logits)

    return logits

  def _quantize(self, inputs, mode):
    # Add noise or quantize (and optionally dequantize in one step).
    half = tf.constant(.5, dtype=self.dtype)
    _, _, channels, input_slices = self._get_input_dims()
    self.active_out_filters = channels

    if mode == "noise":
      noise = tf.random.uniform(tf.shape(inputs), -half, half)
      return tf.math.add_n([inputs, noise])

    medians = self._medians[:self.active_out_filters]
    medians = medians[input_slices]
    outputs = tf.math.floor(inputs + (half - medians))

    if mode == "dequantize":
      outputs = tf.cast(outputs, self.dtype)
      return outputs + medians
    else:
      assert mode == "symbols", mode
      outputs = tf.cast(outputs, tf.int32)
      return outputs

  def _dequantize(self, inputs, mode):
    _, _, _, input_slices = self._get_input_dims()
    medians = self._medians[:self.active_out_filters]
    medians = medians[input_slices]
    outputs = tf.cast(inputs, self.dtype)
    return outputs + medians

  def compress(self, inputs):
    with tf.name_scope(self._name_scope()):
      inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
      if not self.built:
        # Check input assumptions set before layer building, e.g. input rank.
        input_spec.assert_input_compatibility(
            self.input_spec, inputs, self.name)
        if self.dtype is None:
          self._dtype = inputs.dtype.base_dtype.name
        self.build(inputs.shape)

      # Check input assumptions set after layer building, e.g. input shape.
      if not tf.executing_eagerly():
        input_spec.assert_input_compatibility(
            self.input_spec, inputs, self.name)
        if inputs.dtype.is_integer:
          raise ValueError(
              "{} can't take integer inputs.".format(type(self).__name__))

      symbols = self._quantize(inputs, "symbols")
      assert symbols.dtype == tf.int32

      ndim = self.input_spec.ndim
      indexes = self._prepare_indexes(shape=tf.shape(symbols)[1:])
      broadcast_indexes = (indexes.shape.ndims != ndim)
      if broadcast_indexes:
        # We can't currently broadcast over anything else but the batch axis.
        assert indexes.shape.ndims == ndim - 1
        args = (symbols,)
      else:
        args = (symbols, indexes)

      def loop_body(args):
        string = range_coding_ops.unbounded_index_range_encode(
            args[0], indexes if broadcast_indexes else args[1],
            self._quantized_cdf[:self.active_out_filters,:], 
            self._cdf_length[:self.active_out_filters], 
            self._offset[:self.active_out_filters],
            precision=self.range_coder_precision, overflow_width=4,
            debug_level=0)
        return string

      strings = tf.map_fn(
          loop_body, args, dtype=tf.string,
          back_prop=False, name="compress")

      if not tf.executing_eagerly():
        strings.set_shape(inputs.shape[:1])

      return strings

  def sort_weight(self, sess, vst, sorted_idx):
    weights = self.weights
    update_ops = []
    for i in range(len(weights)):
      var_weight = vst[weights[i].name]
      op_weight = tf.assign(var_weight, tf.gather(var_weight, sorted_idx, axis=0))
      update_ops.append(op_weight)
      # matrix,bias,factor,11, quantiles,quantized_cdf,cdf_length
    sess.run(update_ops)

  def sort_weight_graph(self, sorted_idx):
    self._medians = tf.gather(self._medians, sorted_idx, axis=0)
    self._quantized_cdf = tf.gather(self._quantized_cdf, sorted_idx, axis=0)
    self._cdf_length = tf.gather(self._cdf_length, sorted_idx, axis=0)
    self._offset = tf.gather(self._offset, sorted_idx, axis=0)
    for i in range(len(self.filters) + 1):
      self._matrices[i] = tf.gather(self._matrices[i], sorted_idx, axis=0)
      self._biases[i] = tf.gather(self._biases[i], sorted_idx, axis=0)
      if i < len(self._factors):
        self._factors[i] = tf.gather(self._factors[i], sorted_idx, axis=0)
      

class DynamicGaussianConditional(tfc.GaussianConditional):
  def __init__(self, *args, **kwargs):
    super(DynamicGaussianConditional, self).__init__(*args, **kwargs)
    self.active_out_filters = None

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    super(DynamicGaussianConditional, self).build(input_shape)

  
  def compress(self, inputs):
    with tf.name_scope(self._name_scope()):
      inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
      if not self.built:
        # Check input assumptions set before layer building, e.g. input rank.
        input_spec.assert_input_compatibility(
            self.input_spec, inputs, self.name)
        if self.dtype is None:
          self._dtype = inputs.dtype.base_dtype.name
        self.build(inputs.shape)

      # Check input assumptions set after layer building, e.g. input shape.
      if not tf.executing_eagerly():
        input_spec.assert_input_compatibility(
            self.input_spec, inputs, self.name)
        if inputs.dtype.is_integer:
          raise ValueError(
              "{} can't take integer inputs.".format(type(self).__name__))

      symbols = self._quantize(inputs, "symbols")
      assert symbols.dtype == tf.int32

      ndim = self.input_spec.ndim
      indexes = self._prepare_indexes(shape=tf.shape(symbols)[1:])
      broadcast_indexes = (indexes.shape.ndims != ndim)
      if broadcast_indexes:
        # We can't currently broadcast over anything else but the batch axis.
        assert indexes.shape.ndims == ndim - 1
        args = (symbols,)
      else:
        args = (symbols, indexes)

      def loop_body(args):
        string = range_coding_ops.unbounded_index_range_encode(
            args[0], indexes if broadcast_indexes else args[1],
            self._quantized_cdf, 
            self._cdf_length, 
            self._offset,
            precision=self.range_coder_precision, overflow_width=4,
            debug_level=0)
        return string

      #print(symbols, indexes, self._quantized_cdf, self._cdf_length, self._offset)

      strings = tf.map_fn(
          loop_body, args, dtype=tf.string,
          back_prop=False, name="compress")

      if not tf.executing_eagerly():
        strings.set_shape(inputs.shape[:1])

      return strings

  def sort_weight(self, sorted_idx):
    pass


class DynamicGDN(tfc.GDN):
  def __init__(self, *args, **kwargs):
    super(DynamicGDN, self).__init__(*args, **kwargs)
    self.active_out = None

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    super(DynamicGDN, self).build(input_shape)
    self.input_spec = None

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    ndim = self._input_rank

    self.active_out = tf.shape(inputs)[-1]
    gamma = self.gamma[(slice(0,self.active_out), slice(0, self.active_out))]
    beta = self.beta[slice(0, self.active_out)]

    if self.rectify:
      inputs = tf.nn.relu(inputs)

    # Compute normalization pool.
    if ndim == 2:
      norm_pool = tf.linalg.matmul(tf.math.square(inputs), gamma)
      norm_pool = tf.nn.bias_add(norm_pool, beta)
    elif self.data_format == "channels_last" and ndim <= 4:
      # TODO(unassigned): This branch should also work for ndim == 5, but
      # currently triggers a bug in TF.
      #shape = gamma.shape.as_list()
      shape = tf.shape(gamma)
      gamma = tf.reshape(gamma, [1,1, shape[0], shape[1]])
      norm_pool = tf.nn.convolution(tf.math.square(inputs), gamma, "VALID")
      norm_pool = tf.nn.bias_add(norm_pool, beta)
    else:  # generic implementation
      # This puts channels in the last dimension regardless of input.
      norm_pool = tf.linalg.tensordot(
          tf.math.square(inputs), gamma, [[self._channel_axis()], [0]])
      norm_pool += beta
      if self.data_format == "channels_first":
        # Return to channels_first format if necessary.
        axes = list(range(ndim - 1))
        axes.insert(1, ndim - 1)
        norm_pool = tf.transpose(norm_pool, axes)

    if self.inverse:
      norm_pool = tf.math.sqrt(norm_pool)
    else:
      norm_pool = tf.math.rsqrt(norm_pool)
    outputs = inputs * norm_pool

    if not tf.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))
    return outputs

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(input_shape)

  def sort_weight_graph(self, sorted_idx):
    self.beta = tf.gather(self.beta, sorted_idx, axis=0)
    self.gamma = tf.gather(tf.gather(self.gamma, sorted_idx, axis=0), sorted_idx, axis=1)

