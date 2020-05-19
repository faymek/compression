#%%
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

metapath = "tfc_metagraphs/bmshj2018-hyperprior-mse-8.metagraph"

def test():
    with tf.Graph().as_default():
        with tf.io.gfile.GFile(metapath, "rb") as f:
            string = f.read()
        metagraph = tf.MetaGraphDef()
        metagraph.ParseFromString(string)
        tf.train.import_meta_graph(metagraph)
        graph = tf.get_default_graph()

#%%
def board():
    graph = tf.get_default_graph()
    graphdef = graph.as_graph_def()
    _ = tf.train.import_meta_graph(metapath)
    summary_write = tf.summary.FileWriter("./log" , graph)
    summary_write.close()



# %%
with tf.Graph().as_default():
    with tf.io.gfile.GFile(metapath, "rb") as f:
        string = f.read()
    metagraph = tf.MetaGraphDef()
    metagraph.ParseFromString(string)
    tf.train.import_meta_graph(metagraph)
    graph = tf.get_default_graph()

# %%
