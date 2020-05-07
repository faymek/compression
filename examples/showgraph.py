#%%
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

metapath = "tfc_metagraphs/bmshj2018-hyperprior-mse-8.metagraph"

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(metapath)
saver.restore(sess,tf.train.latest_checkpoint('./log'))
# Access the graph
#graph = tf.get_default_graph()
# %%


# %%
