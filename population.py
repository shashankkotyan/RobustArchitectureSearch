#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email:  shashankkotyan@gmail.com
"""

import os, random, pickle, traceback, copy, numpy as np
from itertools import product

import networkx as nx

import build


class Population:


    def set(self, args):

        self.use_adaptive_mutations = args.use_adaptive_mutations
        self.use_cycles             = args.use_cycles
        self.use_random_params      = args.use_random_params
        self.use_non_squares        = args.use_non_squares
        self.log_filepath           = args.log_dir

        self.initial_filter, self.initial_kernel, self.initial_stride  = [8, 16, 32, 64], [1, 3, 5], [1, 2]
        self.initial_units                                             = [64, 128, 256, 512]

        self.initial_layers = list(range(1,2))
        self.initial_blocks = list(range(1,2))

        self.population_layers_count, self.population_blocks_count, self.population_models_count = 0,  0,  0
        self.population_layers,       self.population_blocks,       self.population_models       = [], [], []
        
    
    def write_log(self, string):
        with open(f"{self.log_filepath}/logs.log", "a") as file: file.write(string)

    
    def write_population_log(self, string):
        with open(f"{self.log_filepath}/population.log", "a") as file: file.write(string)
       
    
    def write_create_log(self, string):
        with open(f"{self.log_filepath}/create.log", "a") as file: file.write(string)
    
    
    def write_mutation_log(self, string):
        with open(f"{self.log_filepath}/mutation.log", "a") as file: file.write(string)

    
    def write_exceptions_log(self, string):
        with open(f"{self.log_filepath}/exceptions.log", "a") as file: file.write(string)
        
    
    def read_populations(self, population_dir):
        with open(f"{population_dir}/layers.pkl", 'rb') as file: self.population_layers = pickle.load(file)
        with open(f"{population_dir}/blocks.pkl", 'rb') as file: self.population_blocks = pickle.load(file)
        with open(f"{population_dir}/models.pkl", 'rb') as file: self.population_models = pickle.load(file)

    
    def save_populations(self, population_dir):
        with open(f"{population_dir}/layers.pkl", 'wb') as file: pickle.dump(self.population_layers, file)
        with open(f"{population_dir}/blocks.pkl", 'wb') as file: pickle.dump(self.population_blocks, file)
        with open(f"{population_dir}/models.pkl", 'wb') as file: pickle.dump(self.population_models, file)
        self.write_population_log(f"L:{len(self.population_layers)}, B:{len(self.population_blocks)}, M:{len(self.population_models)}, LC:{self.population_layers_count}, BC:{self.population_blocks_count}, MC:{self.population_models_count},\n")

    
    def update_populations(self, model_dna):

        self.population_models += [model_dna]

        for block_dna in model_dna['constituents']['blocks']:

            self.population_blocks += [block_dna]

            for layer_dna in block_dna['constituents']['layers']:

                self.population_layers += [layer_dna]
    
    
    def clean_populations(self): self.population_layers, self.population_blocks, self.population_models = [], [], []

    
    def create_graph_layer(self, layer):
        
        dna = {}   

        G = nx.DiGraph() 
        G.add_node(f"{self.population_layers_count}")

        parameters = {

            'type_layer':   layer[0],
            'layer_params': layer[1:],            
        }

        for key, value in parameters.items(): 

            nx.set_node_attributes(G, value, key)
            G.graph[key] = value

        dna['graph']        = G
        dna['constituents'] = {'node': layer}
        
        self.population_layers_count += 1

        return dna

    
    def create_graph_block(self, nodes, connections):

        dna = {}   
        in_out   = {}
        vertices = []
        count    = 0

        for i, l in enumerate(nodes):

            in_out[i] = {'start': [ int(n) + count for n, node in enumerate(l['graph'].nodes) if l['graph'].in_degree(node) == 0], 'end': [ int(n) + count for n, node in enumerate(l['graph'].nodes) if l['graph'].out_degree(node) == 0]}
            vertices += [l['graph']]
            count    += len(list(l['graph'].nodes))

        edges = [(start, end) for s, e in connections for start, end in product(in_out[s]['end'], in_out[e]['start'])] 

        G = nx.disjoint_union_all(vertices)
        if len(vertices) == 1: G = nx.convert_node_labels_to_integers(G)
        G.add_edges_from(edges)

        in_out = {'start': [ node for node in G.nodes if G.in_degree(node) == 0], 'end': [ node for node in G.nodes if G.out_degree(node) == 0]}
        
        H = nx.DiGraph() 
        H.add_node(f"input")
        H.add_node(f"output")
        H.nodes['input']['type_layer']  = 'input'
        H.nodes['output']['type_layer'] = 'output'
        
        U = nx.disjoint_union(H, G)
            
        for s in in_out['start']: U.add_edge(0, int(s)+2)
        for e in in_out['end']:   U.add_edge(int(e)+2, 1)

        if len(list(nx.simple_cycles(U))) > 0: raise Exception(f"\nCycle found in the block graph\n")

        # build.check_block(U)
        
        dna['graph']        = G
        dna['constituents'] = {'layers': nodes, 'edges': connections}

        self.population_blocks_count += 1

        return dna
    
    
    def create_graph_model(self, nodes, connections):

        dna = {}
        in_out   = {}
        vertices = []
        count    = 0
        
        for i, l in enumerate(nodes):

            in_out[i] = {'start': [ int(n) + count for n, node in enumerate(l['graph'].nodes) if l['graph'].in_degree(node) == 0], 'end': [ int(n) + count for n, node in enumerate(l['graph'].nodes) if l['graph'].out_degree(node) == 0]}
            vertices += [l['graph']]
            count    += len(list(l['graph'].nodes))

        edges = [(start, end) for s, e in connections for start, end in product(in_out[s]['end'], in_out[e]['start'])] 
        
        G = nx.disjoint_union_all(vertices)
        if len(vertices) == 1: G = nx.convert_node_labels_to_integers(G)
        G.add_edges_from(edges)

        in_out = {'start': [ node for node in G.nodes if G.in_degree(node) == 0], 'end': [ node for node in G.nodes if G.out_degree(node) == 0]}
        
        H = nx.DiGraph() 
        H.add_node(f"input")
        H.add_node(f"output")
        H.nodes['input']['type_layer']  = 'input'
        H.nodes['output']['type_layer'] = 'output'
        
        U = nx.disjoint_union(H,G)
            
        for s in in_out['start']: U.add_edge(0, int(s)+2)
        for e in in_out['end']:   U.add_edge(int(e)+2, 1)

        if len(list(nx.simple_cycles(U))) > 0: raise Exception(f"\nCycle found in the model graph\n")
            
        build.check_block(U)

        parameters = {
            'model_params':  {
                                'weights':       [1/3,1/3,1/3],
                             },
            'spectrum':      self.get_spectrum(U, nodes, connections),
        }
        
        for key, value in parameters.items():  U.graph[key] = value

        dna['graph']        = U
        dna['constituents'] = {'blocks': nodes, 'edges': connections}

        self.update_populations(dna)
        self.population_models_count += 1

        return dna

    
    def create_random_layer(self):

        type_layer = random.choice([0,1])

        if type_layer == 0:

            f, k, s = random.choice(self.initial_filter), random.choice(self.initial_kernel), random.choice(self.initial_stride)
            layer = ['convolution_2d', f, k, k, s, s]
            self.write_create_log(f"Convolution Layer Created Randomly with filter_size ={f} kernel_size={k} stride_size={s}\n")

        elif type_layer == 1:

            u = random.choice(self.initial_units)
            layer = ['fully_connected', u]
            self.write_create_log(f"Dense Layer Created Randomly with units ={u}\n")

        return self.create_graph_layer(layer)

    
    def create_random_block(self):

        n = random.choice(self.initial_layers)

        while True:
            try:
                
                self.write_create_log(f"Block Created Randomly with no of layers ={n}\n")
                return self.create_graph_block([ self.create_random_layer() for _ in range(n)], list(nx.generators.trees.random_tree(n).edges))

            except Exception as e:
               
                self.write_exceptions_log(f"\nException occured in creating block:\n{traceback.format_exc()}\n")

    
    def create_random_model(self):

        n = random.choice(self.initial_blocks)

        while True:
            
            try:
                
                self.write_create_log(f"Model Created Randomly with no of blocks = {n}\n")
                return self.create_graph_model([self.create_random_block() for _ in range(n)], list(nx.generators.trees.random_tree(n).edges))

            except Exception as e:
                
                self.write_exceptions_log(f"\nException occured in creating model:\n{traceback.format_exc()}\n")

    
    def get_random_models(self, n):

        clean_population = self.population_models 

        l = len(clean_population)

        if l > n :

            indices = random.sample(list(range(l)), n)
            models  = [clean_population[index]['constituents'] for index in indices]

        else:

            indices = list(range(l))
            models  = [clean_population[index]['constituents'] for index in indices]

        self.write_create_log(f"\n{n} number of models were retrived and their indices are {indices}\n\n")

        return models


    def get_random_blocks(self, n):
        clean_population = self.population_blocks   
        
        l = len(clean_population)
        
        if l > n:
            
            indices = random.sample(list(range(l)), n)
            blocks  = [clean_population[index] for index in indices]
        
        else:

            indices = list(range(l))
            blocks  = [clean_population[index]['constituents'] for index in indices]

        self.write_create_log(f"\n{n} number of blocks were retrived and their indices are {indices}\n")

        return blocks


    def get_random_layers(self, n, new=False):
        
        clean_population = self.population_layers

        l = len(clean_population)

        if l > n:
            
            indices = random.sample(list(range(l)), n)
            layers  = [clean_population[index] for index in indices]

        else:

            indices = list(range(l))
            layers  = [clean_population[index]['constituents'] for index in indices]

        self.write_create_log(f"\n{n} number of layers were retrived and their indices are {indices}\n")

        return layers


    def mutate(self, old_model):

        mutation = None

        while mutation is None:

            mutated_model = copy.deepcopy(old_model)
            
            if self.use_adaptive_mutations:
                
                mutated_model['graph'].graph['model_params']['weights'] = [ weight + random.uniform(-0.03,0.03)                                for weight in old_model['graph'].graph['model_params']['weights']]
                mutated_model['graph'].graph['model_params']['weights'] = [ float(weight)/sum(mutated_model.graph['model_params']['weights'])  for weight in mutated_model['graph'].graph['model_params']['weights']]

                type_mutate = random.choice([0,1,2], p=mutated_model['graph'].graph['model_params']['weights'])
            
            else:
                
                type_mutate = random.choice([0,1,2]) 

            if type_mutate == 0:   type_mutation, mutation = self.mutate_model(mutated_model)
            elif type_mutate == 1: type_mutation, mutation = self.mutate_block(mutated_model)
            elif type_mutate == 2: type_mutation, mutation = self.mutate_layer(mutated_model)
            
            if mutation is not None:
                
                self.write_mutation_log(f"Mutating model with mutation {type_mutation}\n")
                return type_mutation, mutation


    def mutate_layer(self, old_model):

        convolution_mutation     = {'swap_layer': self.swap_layer,'kernel': self.kernel, 'filter': self.filter, 'stride': self.stride}
        dense_mutation           = {'swap_layer': self.swap_layer,'units':  self.units}

        if self.use_cycles:

            l= len(self.population_layers)

            if l > 200: 
                
                convolution_mutation     = {'swap_layer': self.swap_layer}
                dense_mutation           = {'swap_layer': self.swap_layer}
            
            elif l < 50:
                
                del convolution_mutation['swap_layer']
                del dense_mutation['swap_layer']

        layer_mutation = {'convolution_2d': convolution_mutation, 'fully_connected': dense_mutation }
        
        try:
            
            mutated_model = copy.deepcopy(old_model)

            mutated_block_index = random.choice(list(range(len(mutated_model['constituents']['blocks']))))
            mutated_block       = mutated_model['constituents']['blocks'][mutated_block_index]

            mutated_layer_index = random.choice(list(range(len(mutated_block['constituents']['layers']))))
            mutated_layer       = mutated_block['constituents']['layers'][mutated_layer_index]

            type_mutation       = layer_mutation[mutated_layer['constituents']['node'][0]]
            mutation            = random.choice(list(type_mutation.keys()))

            mutated_layer       = type_mutation[mutation](mutated_layer)

            if mutated_layer is None: return 'Failed', None

            mutated_block['constituents']['layers'][mutated_layer_index] = mutated_layer
            mutated_model['constituents']['blocks'][mutated_block_index] = mutated_block

            try:

                self.create_graph_layer(mutated_layer['constituents']['node'])

            except Exception as e:

                self.write_exceptions_log(f"\nException occured in mutating layer (Creating layer) {mutation}:\n{traceback.format_exc()}\n")
                return 'Failed', None

            try:

                self.create_graph_block(mutated_block['constituents']['layers'], mutated_block['constituents']['edges'])

            except Exception as e:

                self.write_exceptions_log(f"\nException occured in mutating layer (Creating Block) {mutation}:\n{traceback.format_exc()}\n")
                return 'Failed', None

            try:

                mutated = self.create_graph_model(mutated_model['constituents']['blocks'], mutated_model['constituents']['edges'])
                self.write_mutation_log(f"\nSuccessfully Mutated Layer with {mutation}\n")
                return mutation, mutated

            except Exception as e:

                self.write_exceptions_log(f"\nException occured in mutating layer (Creating Model) {mutation}:\n{traceback.format_exc()}\n")
                return 'Failed', None

        except:

            self.write_exceptions_log(f"\nException occured in mutating layer:\n{traceback.format_exc()}\n")
            return 'Failed', None

    
    def mutate_block(self, old_model):

        block_mutation = {  
                            'add_layer':    self.add_layer,    'add_layer_connection':    self.add_layer_connection,
                            'remove_layer': self.remove_layer, 'remove_layer_connection': self.remove_layer_connection,
                            'swap_blocks':  self.swap_blocks
                         }

        if self.use_cycles:

            l= len(self.population_blocks)
            if l > 200:  block_mutation = {'swap_blocks': self.swap_blocks}
            elif l < 50: del block_mutation['swap_blocks']

        try:

            mutated_model = copy.deepcopy(old_model)

            mutated_block_index = random.choice(list(range(len(mutated_model['constituents']['blocks']))))
            mutated_block       = mutated_model['constituents']['blocks'][mutated_block_index]

            type_mutation       = block_mutation
            mutation            = random.choice(list(type_mutation.keys()))

            mutated_block       = type_mutation[mutation](mutated_block)

            if mutated_block is None: return 'Failed', None
            
            mutated_model['constituents']['blocks'][mutated_block_index] = mutated_block

            try:

                self.create_graph_block(mutated_block['constituents']['layers'], mutated_block['constituents']['edges'])

            except Exception as e:

                self.write_exceptions_log(f"\nException occured in mutating block (Creating block) {mutation}:\n{traceback.format_exc()}\n")
                return 'Failed', None

            try:

                mutated = self.create_graph_model(mutated_model['constituents']['blocks'], mutated_model['constituents']['edges'])
                self.write_mutation_log(f"\nSuccessfully Mutated Block with {mutation}\n")
                return mutation, mutated

            except Exception as e:

                self.write_exceptions_log(f"\nException occured in mutating block (Creating Model) {mutation}:\n{traceback.format_exc()}\n")
                return 'Failed', None

        except:

            self.write_exceptions_log(f"\nException occured in mutating block:\n{traceback.format_exc()}\n")
            return 'Failed', None


    def mutate_model(self, old_model):

        model_mutation = {  
                            'add_block': self.add_block,       'add_block_connection': self.add_block_connection,
                            'remove_block': self.remove_block, 'remove_block_connection': self.remove_block_connection 
                         }
        
        try:

            mutated_model       = copy.deepcopy(old_model)

            type_mutation       = model_mutation
            mutation            = random.choice(list(type_mutation.keys()))

            mutated_model       = type_mutation[mutation](mutated_model)

            if mutated_model is None: return 'Failed', None

            try:
                mutated = self.create_graph_model(mutated_model['constituents']['blocks'], mutated_model['constituents']['edges'])
                self.write_mutation_log(f"\nSuccessfully Mutated Model with {mutation}\n")
                
                return mutation, mutated_model
            except Exception as e:
                self.write_exceptions_log(f"\nException occured in mutating model (Creating Model) {mutation}:\n{traceback.format_exc()}\n")
                return 'Failed', None
        
        except:
            self.write_exceptions_log(f"\nException occured in mutating model:\n{traceback.format_exc()}\n")
            return 'Failed', None

    
    def kernel(self, old_layer):

        if   old_layer['constituents']['node'][0] == 'convolution_2d': index = 2 + round(random.random()) if self.use_non_squares else 2
        elif old_layer['constituents']['node'][0] == 'convolution_1d': index = 2
        else: raise Exception('Layer doesnot support kernel mutation')

        mutated_layer = copy.deepcopy(old_layer)
        old_kernel    = old_layer['constituents']['node'][index]

        if self.use_random_params:
            
            new_kernel = random.uniform(old_kernel / 2.0, old_kernel * 2.0)
            new_kernel = int(new_kernel) if int(new_kernel) % 2 != 0 else int(new_kernel) + 1

        else: new_kernel = random.choice([x for x in self.initial_kernel if x != old_kernel])

        mutated_layer['constituents']['node'][index], mutated_layer['constituents']['node'][3] = new_kernel, new_kernel
            
        return mutated_layer

    
    def stride(self, old_layer): 

        if   old_layer['constituents']['node'][0] == 'convolution_2d': index = 4 + round(random.random()) if self.use_non_squares else 4
        elif old_layer['constituents']['node'][0] == 'convolution_1d': index = 3
        else: raise Exception('Layer doesnot support stride mutation')

        mutated_layer = copy.deepcopy(old_layer)
        old_stride    = old_layer['constituents']['node'][index]

        if self.use_random_params: new_stride = round(2** round(random.uniform(old_stride / 2.0, old_stride * 2.0)))
        else:                      new_stride = random.choice([x for x in self.initial_stride if x != old_stride])

        mutated_layer['constituents']['node'][index], mutated_layer['constituents']['node'][5] = new_stride, new_stride

        return mutated_layer

    
    def filter(self, old_layer):

        if old_layer['constituents']['node'][0] != 'convolution_2d' and old_layer['constituents']['node'][0] != 'convolution_1d': raise Exception('Layer doesnot support filter mutation')

        index   = 1

        mutated_layer = copy.deepcopy(old_layer)
        old_filter    = old_layer['constituents']['node'][index]

        if self.use_random_params: new_filter = round(random.uniform(old_filter / 2.0, old_filter * 2.0))
        else:                      new_filter = random.choice([x for x in self.initial_filter if x != old_filter])

        mutated_layer['constituents']['node'][index] = new_filter

        return mutated_layer

    
    def units(self, old_layer):

        if old_layer['constituents']['node'][0] != 'fully_connected': raise Exception('Layer doesnot support units mutation')

        index   = 1

        mutated_layer = copy.deepcopy(old_layer)
        old_units     = old_layer['constituents']['node'][index]

        if self.use_random_params: new_units = round(random.uniform(old_units / 2.0, old_units * 2.0))
        else:                      new_units = random.choice([x for x in self.initial_units if x != old_units])

        mutated_layer['constituents']['node'][index] = new_units

        return mutated_layer

    
    def swap_layer(self, old_layer):

        mutated_layer = old_layer

        while set(mutated_layer['constituents']['node']) == set(old_layer['constituents']['node']): mutated_layer = self.get_random_layers(1)[0]

        return mutated_layer

    
    def swap_blocks(self, old_block):

        mutated_block = old_block

        while set([tuple(layer['constituents']['node']) for layer in mutated_block['constituents']['layers']]) == set([tuple(layer['constituents']['node']) for layer in old_block['constituents']['layers']]): mutated_block = self.get_random_blocks(1)[0]

        return mutated_block

    
    def add_layer(self, old_block):

        mutated_block = copy.deepcopy(old_block)

        try:
            
            l_n     = len(old_block['constituents']['layers'])

            layer   = self.get_random_layers(1)[0]

            mutated_block['constituents']['layers'] += [layer]

            start, end = random.sample(list(range(l_n)) + [-1, -1], 2)

            if start == -1 and end != -1:
                
                mutated_block['constituents']['edges'] += [(l_n, end)]
            
            elif start != -1 and end == -1:
                
                mutated_block['constituents']['edges'] += [(start, l_n)]
            
            elif start != -1 and end != -1:
                
                mutated_block['constituents']['edges'] += [(start, l_n)]
                mutated_block['constituents']['edges'] += [(l_n, end)]

            return mutated_block
       
        except Exception as e:
            
            self.write_exceptions_log(f"\nException occured in mutating block (Add Layer):\n{traceback.format_exc()}\n")
            return None

    
    def add_layer_connection(self, old_block):

        mutated_block = copy.deepcopy(old_block)

        l = len(mutated_block['constituents']['layers'])

        if l > 2:
            try:
                
                new_connection = mutated_block['constituents']['edges'][0]
                while new_connection in mutated_block['constituents']['edges']: new_connection = tuple(random.sample(list(range(l)), 2))
                
                mutated_block['constituents']['edges'] += [new_connection]
                return mutated_block
            
            except Exception as e:
                
                self.write_exceptions_log(f"\nException occured in mutating block (Add Layer Connection):\n{traceback.format_exc()}\n")
                return None
        
        return None

    
    def remove_layer(self, old_block):

        mutated_block = copy.deepcopy(old_block)

        l = len(mutated_block['constituents']['layers'])

        if l > 2:

            try:  

                mutated_layer_index = random.choice(list(range(l)))
                mutated_layer       = mutated_block['constituents']['layers'][mutated_layer_index]

                del mutated_block['constituents']['layers'][mutated_layer_index]

                mutated_connections = []
                for i, edge in enumerate(mutated_block['constituents']['edges']):
                    
                    if edge[0] == mutated_layer_index or edge[1] == mutated_layer_index: del mutated_block['constituents']['edges'][i]
                    
                    else:
                        
                        new_edge = []
                        
                        for i in range(2):
                            
                            if edge[i] > mutated_layer_index: new_edge += [edge[i] - 1]
                            else:                             new_edge += [edge[i]]
                        
                        mutated_connections.append(tuple(new_edge))

                mutated_block['constituents']['edges'] = mutated_connections

                return mutated_block
           
            except Exception as e:
                
                self.write_exceptions_log(f"\nException occured in mutating block (Remove Layer):\n{traceback.format_exc()}\n")
                return None

        return None

    
    def remove_layer_connection(self, old_block):

        mutated_block = copy.deepcopy(old_block)

        l = len(mutated_block['constituents']['edges'])

        if l > 1:

            try:
                
                mutated_connection_index = random.choice(list(range(l)))
                mutated_connection       = mutated_block['constituents']['edges'][mutated_connection_index]
                
                del mutated_block['constituents']['edges'][mutated_connection_index]

                return mutated_block
            
            except Exception as e:
                
                self.write_exceptions_log(f"\nException occured in mutating block (Remove Layer Connection):\n{traceback.format_exc()}\n")
                return None

        return None

    
    def add_block(self, old_model):

        mutated_model = copy.deepcopy(old_model)

        try:

            l_n     = len(old_model['constituents']['blocks'])

            block   = self.get_random_blocks(1)[0]

            mutated_model['constituents']['blocks'] += [block]

            start, end = random.sample(list(range(l_n)) + [-1, -1], 2)

            if start == -1 and end != -1:

                mutated_model['constituents']['edges'] += [(l_n, end)]
            
            elif start != -1 and end == -1:

                mutated_model['constituents']['edges'] += [(start, l_n)]
            
            elif start != -1 and end != -1:

                mutated_model['constituents']['edges'] += [(start, l_n)]
                mutated_model['constituents']['edges'] += [(l_n, end)]

            return mutated_model

        except Exception as e:
            
            self.write_exceptions_log(f"\nException occured in mutating model (Add Block):\n{traceback.format_exc()}\n")
            return None

    
    def add_block_connection(self, old_model):

        mutated_model = copy.deepcopy(old_model)

        l = len(mutated_model['constituents']['blocks'])
        
        if l > 2:
        
            try:

                new_connection = mutated_model['constituents']['edges'][0]
                while new_connection in mutated_model['constituents']['edges']: new_connection = tuple(random.sample(list(range(l)), 2))

                mutated_model['constituents']['edges'] += [new_connection]
                
                return mutated_model

            except Exception as e:

                self.write_exceptions_log(f"\nException occured in mutating model (Add Block Connection):\n{traceback.format_exc()}\n")
                return None 

        return None

    
    def remove_block(self, old_model):

        mutated_model = copy.deepcopy(old_model)

        l = len(old_model['constituents']['blocks'])

        if l > 2:

            try:

                mutated_block_index = random.choice(list(range(l)))
                mutated_block       = mutated_model['constituents']['blocks'][mutated_block_index]

                del mutated_model['constituents']['blocks'][mutated_block_index]

                mutated_connections =[]
                for i, edge in enumerate(mutated_model['constituents']['edges']):

                    if edge[0] == mutated_block_index or edge[1] == mutated_block_index: del mutated_model['constituents']['edges'][i]
                    else:
                        
                        new_edge = []

                        for i in range(2):
                            
                            if edge[i] > mutated_block_index: new_edge += [edge[i] - 1]
                            else:                             new_edge += [edge[i]]

                        mutated_connections.append(tuple(new_edge))

                mutated_model['constituents']['edges'] = mutated_connections

                return mutated_model

            except Exception as e:

                self.write_exceptions_log(f"\nException occured in mutating model (Remove Block):\n{traceback.format_exc()}\n")
                return None

        return None

    
    def remove_block_connection(self, old_model):

        mutated_model = copy.deepcopy(old_model)

        l = len(mutated_model['constituents']['edges'])

        if l > 1:

            try:

                mutated_connection_index = random.choice(list(range(l)))
                mutated_connection       = mutated_model['constituents']['edges'][mutated_connection_index]
                
                del mutated_model['constituents']['edges'][mutated_connection_index]

                return mutated_model

            except Exception as e:

                self.write_exceptions_log(f"\nException occured in mutating model (Remove Block Connection):\n{traceback.format_exc()}\n")
                return None

        return None

   
    def get_spectrum(self, graph, blocks, block_connections):
        
        layers                 = graph.nodes
        layer_connections      = graph.edges

        num_blocks             = len(blocks)
        num_block_connections  = len(block_connections)
        
        num_total_layers       = len(list(layers))
        num_total_connections  = len(list(layer_connections))

        num_d_layers, num_c_layers = 0, 0

        for layer in layers:

            if   layers[layer]['type_layer'] == 'fully_connected': num_d_layers += 1
            elif layers[layer]['type_layer'] == 'convolution_2d':  num_c_layers += 1

        num_dd_connection, num_dc_connection, num_cd_connection, num_cc_connection = 0, 0, 0, 0

        for start, end in layer_connections:

            t_start = layers[start]['type_layer']
            t_end   = layers[end]['type_layer']

            if t_start == 'fully_connected':

                if   t_end == 'fully_connected': num_dd_connection += 1
                elif t_end == 'convolution_2d':  num_dc_connection += 1

            elif t_start == 'convolution_2d':
                
                if   t_end == 'fully_connected': num_cd_connection += 1
                elif t_end == 'convolution_2d':  num_cc_connection += 1

        return [
                num_blocks,   num_block_connections,
                num_d_layers, num_dd_connection, num_dc_connection,
                num_c_layers, num_cd_connection, num_cc_connection,
                num_total_layers, num_total_connections,
                ]