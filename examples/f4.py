#%%
"""
bmshj2018

"""

import argparse
import glob
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc
from dynamic import *


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, lmbda, *args, **kwargs):
    self.num_filters = num_filters
    self.lmbda = lmbda
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True, 
            activation=tfc.GDN(name="gdn_0")),
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True, 
            activation=tfc.GDN(name="gdn_1")),
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True, 
            activation=tfc.GDN(name="gdn_2")),
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True, 
            activation=None),
    ]
    super(AnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor, self.lmbda)
    return tensor


class SynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, lmbda, *args, **kwargs):
    self.num_filters = num_filters
    self.lmbda = lmbda
    super(SynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, 
            activation=tfc.GDN(name="igdn_0", inverse=True)),
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, 
            activation=tfc.GDN(name="igdn_1", inverse=True)),
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, 
            activation=tfc.GDN(name="igdn_2", inverse=True)),
        CondSignalConv2D(
            3, (5, 5), name="layer_3", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, 
            activation=None),
    ]
    super(SynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor, self.lmbda)
    return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters, lmbda, *args, **kwargs):
    self.num_filters = num_filters
    self.lmbda = lmbda
    super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        CondSignalConv2D(
            self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
            padding="same_zeros", use_bias=True, 
            activation=tf.nn.relu),
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True, 
            activation=tf.nn.relu),
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=False, 
            activation=None),
    ]
    super(HyperAnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor, self.lmbda)
    return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters, lmbda, *args, **kwargs):
    self.num_filters = num_filters
    self.lmbda = lmbda
    super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, 
            kernel_parameterizer=None, activation=tf.nn.relu),
        CondSignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, 
            kernel_parameterizer=None, activation=tf.nn.relu),
        CondSignalConv2D(
            self.num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
            padding="same_zeros", use_bias=True, 
            kernel_parameterizer=None, activation=None),
    ]
    super(HyperSynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor, self.lmbda)
    return tensor


def test_train(args):
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device("/cpu:0"):
    train_files = glob.glob(args.train_glob)
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  lmbda_level = tf.random_uniform([], minval=0, maxval = 8, dtype=tf.int32)
  lmbda_onehot = tf.one_hot(tf.reshape(lmbda_level,[1]), depth=8)
  lmbda = 0.1 * tf.pow(2.0, tf.cast(lmbda_level, tf.float32) - 6.0)

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters, lmbda_onehot)
  synthesis_transform = SynthesisTransform(args.num_filters, lmbda_onehot)
  hyper_analysis_transform = HyperAnalysisTransform(args.num_filters, lmbda_onehot)
  hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters, lmbda_onehot)
  entropy_bottleneck = tfc.EntropyBottleneck()

  # Build autoencoder and hyperprior.
  y = analysis_transform(x)
  z = hyper_analysis_transform(abs(y))
  z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
  sigma = hyper_synthesis_transform(z_tilde)
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
  y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde)

  # Total number of bits divided by number of pixels.
  train_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
               tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2

  # The rate-distortion cost.
  train_loss = lmbda * train_mse + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse)
  tf.summary.scalar("lambda", lmbda_level)

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  hooks = [
    tf.train.StopAtStepHook(last_step=args.last_step),
    tf.train.NanTensorHook(train_loss),
  ]
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_secs=60) as sess:
    while not sess.should_stop():
      sess.run(train_op)
      

def test_compress(args):
  """Compresses an image."""

  # Load input image and add batch dimension.
  fn = tf.placeholder(tf.string, [])

  x = read_png(fn)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])
  x_shape = tf.shape(x)

  lmbda_level = tf.placeholder(tf.int32, [])
  lmbda_onehot = tf.one_hot(tf.reshape(lmbda_level,[1]), depth=8)
  lmbda = 0.1 * tf.pow(2.0, tf.cast(lmbda_level, tf.float32) - 6.0)

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters, lmbda_onehot)
  synthesis_transform = SynthesisTransform(args.num_filters, lmbda_onehot)
  hyper_analysis_transform = HyperAnalysisTransform(args.num_filters, lmbda_onehot)
  hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters, lmbda_onehot)
  entropy_bottleneck = tfc.EntropyBottleneck()

  # Transform and compress the image.
  y = analysis_transform(x)
  y_shape = tf.shape(y)
  z = hyper_analysis_transform(abs(y))
  z_hat, z_likelihoods = entropy_bottleneck(z, training=False)
  sigma = hyper_synthesis_transform(z_hat)
  sigma = sigma[:, :y_shape[1], :y_shape[2], :]
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
  side_string = entropy_bottleneck.compress(z)
  string = conditional_bottleneck.compress(y)

  # Transform the quantized image back (if requested).
  y_hat, y_likelihoods = conditional_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat)
  x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
              tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  x *= 255
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)

  mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
  psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)

    f = open("f4.csv", "w")
    print("level, fn, bpp, mse, np", file=f)
    for i in np.arange(0,8):
      for filename in glob.glob("kodak/*.png"):

        v_lmbda_level, v_eval_bpp, v_mse, v_num_pixels = sess.run(
            [lmbda_level, eval_bpp, mse, num_pixels], feed_dict={fn: filename, lmbda_level: i})

        print("%.2f, %s, %.4f, %.4f, %d"%(v_lmbda_level, filename, v_eval_bpp, v_mse, v_num_pixels), file=f)
    f.close()  



def test_compress_mask(args):
  """Compresses an image."""

  # Load input image and add batch dimension.
  fn = tf.placeholder(tf.string, [])

  x = read_png(fn)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])
  x_shape = tf.shape(x)

  lmbda_level = tf.placeholder(tf.int32, [])
  lmbda_onehot = tf.one_hot(tf.reshape(lmbda_level,[1]), depth=8)
  lmbda = 0.1 * tf.pow(2.0, tf.cast(lmbda_level, tf.float32) - 6.0)

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters, lmbda_onehot)
  synthesis_transform = SynthesisTransform(args.num_filters, lmbda_onehot)
  hyper_analysis_transform = HyperAnalysisTransform(args.num_filters, lmbda_onehot)
  hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters, lmbda_onehot)
  entropy_bottleneck = tfc.EntropyBottleneck()

  # Transform and compress the image.
  y = analysis_transform(x)
  y_shape = tf.shape(y)
  z = hyper_analysis_transform(abs(y))
  z_hat, z_likelihoods = entropy_bottleneck(z, training=False)
  sigma = hyper_synthesis_transform(z_hat)
  sigma = sigma[:, :y_shape[1], :y_shape[2], :]
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
  side_string = entropy_bottleneck.compress(z)
  string = conditional_bottleneck.compress(y)

  # Transform the quantized image back (if requested).
  y_hat, y_likelihoods = conditional_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat)
  x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
              tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  x *= 255
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)

  mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
  psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)

    tn_iter = {
      "trans": ["analysis", "hyper_analysis", "hyper_synthesis", "synthesis"],
      "layers": [4,3,3,3],
      "ops": ["fc_u"]
    }

    tn_names = ["{}_transform/layer_{}/{}/Softplus:0".format(tran,ln,op) \
                  for tran, layer in zip(tn_iter["trans"], tn_iter["layers"]) \
                  for ln in range(layer) for op in tn_iter["ops"]]
    print(tn_names)

    import pandas as pd

    df = pd.DataFrame()

    for i in np.arange(0,8):
      for name in tn_names:
        tn = tf.get_default_graph().get_tensor_by_name(name)
        tnv = sess.run(tf.reshape(tn, [256]), feed_dict={lmbda_level: i})
        df1 = pd.DataFrame({"level":i, "name":name, "width":range(256), "value":tnv})
        df = df.append(df1)
    df.to_csv("mask.csv") 



def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="train",
      help="Directory where to save/load model checkpoints.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model.")
  train_cmd.add_argument(
      "--train_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help="Output filename (optional). If not provided, appends '{}' to "
             "the input filename.".format(ext))

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    test_train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    test_compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    test_decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
