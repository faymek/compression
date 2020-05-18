import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
from tensorflow_compression.python.layers import parameterizers
from tensorflow_compression.python.ops import padding_ops
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import range_coding_ops
from tensorflow.python.keras.engine import input_spec

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

  def sort_filter(self, sorted_idx, sort_out=True):
    import numpy as np
    # sort in_channel by input idx from input layer
    if sorted_idx is not None:
      self._kernel = tf.gather(self._kernel, sorted_idx, axis=2)
    # sort out_channel by calulate L1 norm
    if sort_out:
      importance = tf.reduce_sum(tf.abs(self.kernel), axis=[0,1,2])
      endpoint = self.kernel.shape.as_list()[3]-self.active_out_filters
      if endpoint > 0:
        protect = tf.range(0, -endpoint, -1.0)
        importance = tf.concat([importance[:self.active_out_filters], protect], axis=0)
      sorted_idx = tf.argsort(importance, direction='DESCENDING')
      self._kernel = tf.gather(self._kernel, sorted_idx, axis=3)
      if self.use_bias:
        self._bias = tf.gather(self._bias, sorted_idx, axis=0)
      if isinstance(self.activation, DynamicGDN):
        self.activation.sort_weight(sorted_idx)
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
            self._quantized_cdf[:self.active_out_filters], 
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


  def sort_weight(self, sorted_idx):
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
            self._quantized_cdf[:self.active_out_filters], 
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


  def sort_weight(self, sorted_idx):
    self._quantized_cdf = tf.gather(self._quantized_cdf, sorted_idx, axis=0)
    self._cdf_length = tf.gather(self._cdf_length, sorted_idx, axis=0)
    self._offset = tf.gather(self._offset, sorted_idx, axis=0)


class DynamicGDN(tfc.GDN):
  def __init__(self, *args, **kwargs):
    super(DynamicGDN, self).__init__(*args, **kwargs)
    self.active_out_filters = None

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    super(DynamicGDN, self).build(input_shape)
    self.input_spec = None

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    ndim = self._input_rank

    input_shape = inputs.shape
    channel_axis = {"channels_first": 1, "channels_last": -1}[self.data_format]
    self.active_out_filters = input_shape.as_list()[channel_axis]
    gamma = self.gamma[(slice(0,self.active_out_filters), slice(0, self.active_out_filters))]
    beta = self.beta[slice(0, self.active_out_filters)]

    if self.rectify:
      inputs = tf.nn.relu(inputs)

    # Compute normalization pool.
    if ndim == 2:
      norm_pool = tf.linalg.matmul(tf.math.square(inputs), gamma)
      norm_pool = tf.nn.bias_add(norm_pool, beta)
    elif self.data_format == "channels_last" and ndim <= 4:
      # TODO(unassigned): This branch should also work for ndim == 5, but
      # currently triggers a bug in TF.
      shape = gamma.shape.as_list()
      gamma = tf.reshape(gamma, (ndim - 2) * [1] + shape)
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

  def sort_weight(self, sorted_idx):
    self.gamma = tf.gather(tf.gather(self.gamma, sorted_idx, axis=0) ,sorted_idx, axis=1)
    self.beta = tf.gather(self.beta, sorted_idx, axis=0)

