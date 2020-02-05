#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email:  shashankkotyan@gmail.com
"""

import os, random, traceback, pickle, numpy as np
from time import time

import build
from process import Process
      
       
class Worker:

    
    def __init__(self, args, data):
        
        self.test              = args.test

        self.dataset          = args.dataset
        self.use_augmentation = args.use_augmentation
        self.use_limited_data = args.use_limited_data
        self.epochs           = args.epochs

        self.data = data
        self.batch_size = 128
    
    
    def preprocess(self, x):

        def process(x, mean, std): 
            for i in range(3): x[:,:,:,i] = (x[:,:,:,i] - mean[i]) / std[i]
            return x

        if self.dataset == 2: mean, std = [125.307, 122.95, 113.865], [62.9932, 62.0887, 66.7048]
        else:                 mean, std = [0, 0, 0], [255, 255, 255]

        return process(x, mean, std)
    
    
    def train(self, gen):
        
        self.x_train, self.y_train, self.x_test, self.y_test = self.preprocess(self.data['x_train']), self.data['y_train'], self.preprocess(self.data['x_test']), self.data['y_test']
        
        if self.test: history = self.train_test()

        else:

            if self.use_limited_data: history = self.train_limited()
            else:                     history = self.train_all()

        return history.history
    
    
    def train_test(self):

        return self.model.fit(
                self.x_train[:1], self.y_train[:1], batch_size=self.batch_size, 
                epochs=1, verbose=0, validation_data=(self.x_test[:1], self.y_test[:1])
                )

    
    def train_all(self):

        if self.use_augmentation:

            return self.model.fit_generator(
                self.data['datagen'].flow(self.x_train, self.y_train, batch_size=self.batch_size), 
                steps_per_epoch= (len(self.data['x_train'])//self.batch_size), 
                epochs=self.epochs, verbose=0, callbacks=self.data['callbacks'], validation_data=(self.x_test, self.y_test)
            )

        else:

            return self.model.fit(
                self.x_train, self.y_train, batch_size=self.batch_size, 
                epochs=self.epochs, verbose=0, callbacks=self.data['callbacks'], validation_data=(self.x_test, self.y_test)
            )
            
    
    def train_limited(self):

        random.seed(time())

        indices = random.sample(list(range(self.data['count_x_train'])), 0.1*self.data['count_x_train'])

        if self.use_augmentation:

            return self.model.fit_generator(
                self.data['datagen'].flow(self.x_train[indices], self.y_train[indices], batch_size=self.batch_size), steps_per_epoch= (len(self.data['x_train'][:10000])//self.batch_size), 
                epochs=self.epochs, verbose=0, callbacks=self.data['callbacks'], validation_data=(self.x_test, self.y_test)
                )

        else:

            return self.model.fit(
                self.x_train[indices], self.y_train[indices], batch_size=self.batch_size, 
                epochs=self.epochs, verbose=0, callbacks=self.data['callbacks'], validation_data=(self.x_test, self.y_test)
                )
                    
    
    def run_model(self, gpu_index, individual, dna, gen):
        
        self.model = build.build_block(dna['graph'], num_classes=self.data['num_classes'], gpu_index=gpu_index)
        
        start_time          = time() 
        history             = self.train(gen)
        end_time            = time()

        with open(f"{individual}/training_history.pkl", 'wb')  as history_file:  pickle.dump(history, history_file, pickle.HIGHEST_PROTOCOL)
        
        fitness = history['val_accuracy'][-1]
        metrics = {"fitness": fitness, "evaluation_time": end_time - start_time}
        
        with open(f"{individual}/metrics.pkl", 'wb')           as metrics_file:  pickle.dump(metrics, metrics_file, pickle.HIGHEST_PROTOCOL)
        
        # from tensorflow.keras import utils
        # utils.plot_model(self.model, show_shapes=True, to_file=f"{individual}/model.png")

    
    def evaluate_individual(self, gpu_index, individual, dna, gen, population):

        try:

            p = Process(target=self.run_model, args=(gpu_index, individual, dna, gen))
            p.start()
            p.join()
            if p.exception is not None: raise Exception(f"{p.exception[0]}, {p.exception[1]}")

            population.update_populations(dna)
            with open(f"{individual}/metrics.pkl", 'rb')  as metrics_file: metrics = pickle.load(metrics_file)

            population.write_log(f"GPU {gpu_index} completed training {individual.split('/')[2]} in {metrics['evaluation_time']:.2f} seconds with fitness {metrics['fitness']:.2f}\n")
            
            return metrics

        except Exception as e:

            population.write_exceptions_log(f"Exception occured at training model: {e} \n{traceback.format_exc()}\n")
            return None

    
    def create_child(self, gpu_index, parent_dna, num_mutations, individual, population, store_individual, gen):

        metrics = None
        while metrics is None:

            dna = parent_dna
            mutations  = []

            for _ in range(num_mutations):

                test_dna = None
                while test_dna is None: mutation, test_dna = population.mutate(dna)

                dna        = test_dna
                mutations += [mutation]

            metrics  = self.evaluate_individual(gpu_index, individual, dna, gen, population)
            
        store_individual(individual, dna, metrics, mutations)
        open(f"{individual}/alive.txt", 'w').close()
            
    
    def create_parent(self, gpu_index, individual, population, store_individual, gen):

        metrics = None

        while metrics is None:

            dna      = population.create_random_model()
            metrics  = self.evaluate_individual(gpu_index, individual, dna, gen, population)

        store_individual(individual, dna, metrics)
        open(f"{individual}/alive.txt", 'w').close()