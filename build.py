#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email:  shashankkotyan@gmail.com
"""

import os, traceback, random, numpy as np

import networkx as nx
from matplotlib import pyplot as plt

from process import Process

x = np.zeros((1,32,32,3))
y = np.zeros((1,10))
y[0] = 1


def plot_graph(individual, dna):

    G= dna['graph']

    dense_nodes,       conv_nodes,      mandatory_nodes = [], [], []
    dense_nodes_color, conv_nodes_color                 = [], []

    cmap = plt.cm.get_cmap('Blues', 4)
    dmap = plt.cm.get_cmap('Blues', 4)

    for node in G.nodes:

        layer_graph  = G.nodes[node]

        if layer_graph['type_layer'] == 'fully_connected':

            dense_nodes += [node]

            if       layer_graph['use_bn'] and     layer_graph['use_dropout'][0]: dense_nodes_color += [dmap(1)]
            elif     layer_graph['use_bn'] and not layer_graph['use_dropout'][0]: dense_nodes_color += [dmap(2)]
            elif not layer_graph['use_bn'] and     layer_graph['use_dropout'][0]: dense_nodes_color += [dmap(3)]
            else:                                                                 dense_nodes_color += [dmap(4)]

        elif layer_graph['type_layer'] == 'convolution_2d':

            conv_nodes  += [node]

            if       layer_graph['use_bn'] and     layer_graph['use_dropout'][0]: conv_nodes_color += [cmap(1)]
            elif     layer_graph['use_bn'] and not layer_graph['use_dropout'][0]: conv_nodes_color += [cmap(2)]
            elif not layer_graph['use_bn'] and     layer_graph['use_dropout'][0]: conv_nodes_color += [cmap(3)]
            else:                                                                 conv_nodes_color += [cmap(4)]

        else:

            mandatory_nodes += [node] 
    
    layout = ['random', 'circular', 'shell', 'spring', 'spectral', 'kamada_kawai', 'planar', 'spiral']

    for i in range(len(layout)):

        try:

            if i == 0:   pos = nx.random_layout(G)
            elif i == 1: pos = nx.circular_layout(G)
            elif i == 2: pos = nx.shell_layout(G)
            elif i == 3: pos = nx.spring_layout(G)
            elif i == 4: pos = nx.spectral_layout(G)
            elif i == 5: pos = nx.kamada_kawai_layout(G)
            elif i == 6: pos = nx.planar_layout(G)
            elif i == 7: pos = nx.spiral_layout(G)
            
            nx.draw_networkx_nodes(G, pos, nodelist=dense_nodes,     node_size=300, node_color=dense_nodes_color, node_shape='o', cmap=dmap)
            nx.draw_networkx_nodes(G, pos, nodelist=conv_nodes,      node_size=300, node_color=conv_nodes_color,  node_shape='s', cmap=cmap)
            nx.draw_networkx_nodes(G, pos, nodelist=mandatory_nodes, node_size=300, node_color='black',           node_shape='h')
            nx.draw_networkx_edges(G, pos)

            plt.savefig(f"{individual}/network_{layout[i]}.png", bbox='tight', dpi=300)
            plt.clf()
            
        except Exception as e:

            pass


def build_block(G, num_classes=1, gpu_index=None):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
    if gpu_index is not None: os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    else:                     os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
    import tensorflow as tf
    
    if gpu_index is not None: 

        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:

            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            except RuntimeError as e: print(e)

    from tensorflow.keras import callbacks, datasets, utils, layers, models, optimizers, backend as K
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    keras_layers = {}

    input_node, output_node = 0, 1
    keras_layers[input_node]  = layers.Input(shape=(32,32,3))
    keras_layers[output_node] = layers.GlobalAveragePooling2D()

    for node in G.nodes:

        layer_graph  = G.nodes[node]

        if layer_graph['type_layer'] in ['fully_connected', 'convolution_2d']:

            layer_params = layer_graph['layer_params']
            
            if layer_graph['type_layer'] == 'convolution_2d':     layer = layers.Conv2D(filters=layer_params[0], kernel_size=(layer_params[1], layer_params[2]), strides=(layer_params[3], layer_params[4]), padding='same') 
            elif layer_graph['type_layer'] == 'fully_connected':  layer = layers.Dense(units=layer_params[0])

            keras_layers[node] = layer
    
    explored, queue = [input_node], [input_node]

    while queue:
        
        node = queue.pop(0)

        for successor in G.successors(node): 
            add = True
            for predecessor in G.predecessors(successor):
                if predecessor not in explored: add = False
            if add: queue.append(successor)

        if node not in explored:
            
            predecessors = [predecessor for predecessor in G.predecessors(node)]
            
            if len(predecessors) == 1: 

                if predecessors[0] == 0: ilayer = layers.Lambda(lambda x: x)(keras_layers[predecessors[0]])
                else:                    ilayer = keras_layers[predecessors[0]]

            elif len(predecessors) > 1:

                ilayer = layers.Concatenate()([layers.Flatten()(keras_layers[predecessor]) for predecessor in predecessors])
                
                shape  = ilayer.shape[1]

                if shape % 65536   == 0: ilayer = layers.Reshape((256, 256, shape // 65536))(ilayer)
                elif shape % 16384 == 0: ilayer = layers.Reshape((128, 128, shape // 16384))(ilayer)
                elif shape % 4096  == 0: ilayer = layers.Reshape((64, 64,   shape // 4096))(ilayer)
                elif shape % 1024  == 0: ilayer = layers.Reshape((32, 32,   shape // 1024))(ilayer)
                elif shape % 256   == 0: ilayer = layers.Reshape((16, 16,   shape // 256))(ilayer)
                elif shape % 64    == 0: ilayer = layers.Reshape((8, 8,     shape // 64))(ilayer)
                elif shape % 16    == 0: ilayer = layers.Reshape((4, 4,     shape // 16))(ilayer)
                elif shape % 4     == 0: ilayer = layers.Reshape((2, 2,     shape // 4))(ilayer)
                else:                    ilayer = layers.Reshape((1, 1,     shape))(ilayer)

            keras_layers[node] = keras_layers[node](ilayer)

            layer_graph = G.nodes[node]
            
            if layer_graph['type_layer'] != 'output':

                keras_layers[node] = layers.BatchNormalization()(keras_layers[node])
                keras_layers[node] = layers.Activation('relu')(keras_layers[node])

            explored.append(node)

            for successor in G.successors(input_node): 
                add = True
                for predecessor in G.predecessors(successor):
                     if predecessor not in explored: add = False
                if add: queue.append(successor)
                else:   waiting.append(successor)
    
    assert set(explored) == set(list(G.nodes))
        
    layer = layers.Dense(units=10, activation='softmax', name='Output')(keras_layers[output_node])
                
    model = models.Model(inputs=keras_layers[input_node], outputs=layer)
    model.compile(optimizer=optimizers.Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])
        
    try:                    model.fit(x, y, epochs=1, verbose=0)
    except Exception as e : raise Exception(f"Model cannot be trained: {e} \n{traceback.format_exc()}\n")

    return model


def check_block(graph):

    p = Process(target=build_block, args=(graph, 1, None))
    p.start()
    p.join()
    if p.exception is not None: raise Exception(f"{p.exception[0]}, {p.exception[1]}")


def build_keras_block(graph,individual, fitness_string):

    from tensorflow.keras.utils import plot_model
    plot_model(build_block(graph, 1, None), to_file=f"{individual}/keras_graph_{fitness_string}.png", dpi=300)
    

def check_keras_block(graph, individual, fitness_string):

    p = Process(target=build_keras_block, args=(graph, individual, fitness_string))
    p.start()
    p.join()
    if p.exception is not None: raise Exception(f"{p.exception[0]}, {p.exception[1]}")


def load_data(dataset):

    from tensorflow.keras import callbacks, datasets, utils
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    def preprocess(x): 
        
        for i in range(3): x[:,:,:,i] = (x[:,:,:,i] - mean[i]) / std[i]
        return x 

    if dataset == 0:

        num_classes = 10
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train, x_test = x_train.reshape(-1, 28, 28, 1).astype('float32'), x_test.reshape(-1, 28, 28, 1).astype('float32')
        y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10)

    elif dataset == 1:

        num_classes = 10
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
        x_train, x_test = x_train.reshape(-1, 28, 28, 1).astype('float32'), x_test.reshape(-1, 28, 28, 1).astype('float32')
        y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10)

    elif dataset == 2:

        num_classes = 10
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
        y_train, y_test = utils.to_categorical(y_train[:,0], 10), utils.to_categorical(y_test[:,0], 10)
    
    elif dataset == 3:

        num_classes = 100
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
        mean, std = [125.307, 122.95, 113.865], [62.9932, 62.0887, 66.7048]
        x_train, x_test = preprocess(x_train.astype('float32')), preprocess(x_test.astype('float32'))
        y_train, y_test = utils.to_categorical(y_train[:,0], 100), utils.to_categorical(y_test[:,0], 100)
    
    cbks  = []
    cbks += [callbacks.EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=15)]

    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)
    datagen.fit(x_train)

    return {
            'callbacks':cbks,   'datagen': datagen, 
            'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test,
            'count_train': len(x_train), 'count_test': len(x_test), 
            'num_classes': num_classes
           }
