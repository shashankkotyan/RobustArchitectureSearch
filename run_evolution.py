#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email:  shashankkotyan@gmail.com
"""

import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')

import argparse, glob, pickle, time, GPUtil, numpy as np 

from scipy.cluster.vq import whiten as normalise
from multiprocessing.managers import BaseManager

import build
from population import Population
from worker import Worker
from process import Process


class NeuroEvolution:


    def __init__(self, args):
   
        self.population_size = args.population_size
        self.population_dir  = args.population_dir
        self.num_mutations   = args.num_mutations
        self.number_of_child = args.num_mutated_child
         
        self.gen         = -1
        self.ind_counter = 0

        if not os.path.exists(self.population_dir): os.makedirs(self.population_dir)
        if not os.path.exists(args.log_dir):        os.makedirs(args.log_dir)

        BaseManager.register('Population', Population)
        manager = BaseManager()
        manager.start()
       
        self.population = manager.Population()
        self.population.set(args)
        
        self.worker = Worker(args, build.load_data(args.dataset))

    
    def parallel_run(self, type_create, args):

        start = False

        while not start:

            for deviceID in range(7):

                if deviceID in self.jobs:

                    if not self.jobs[deviceID].is_alive():

                        self.jobs[deviceID].close() 
                        if self.jobs[deviceID].exception is not None: raise Exception(f"{self.jobs[deviceID].exception[0]}, {self.jobs[deviceID].exception[1]}")
                        finished_job = self.jobs.pop(deviceID, None)

            with open("num_workers.txt", "r") as f: num_workers = int(f.read())

            deviceIDs = GPUtil.getAvailable(order='memory', limit=7, maxLoad=1.1, maxMemory=0.5)

            alive = -1

            if len(deviceIDs) != 0:

                for deviceID in deviceIDs:

                    if deviceID not in self.jobs:

                        alive = deviceID
                        break

            if len(self.jobs) < num_workers and alive > -1:

                print(f"GPU {alive} running {self.ind_counter}")

                if type_create == 0: target=self.worker.create_parent
                else:                target=self.worker.create_child

                start = True

                args[0]          = alive
                job              = Process(target=target, args=tuple(args))
                self.jobs[alive] = job 
                job.start()

            else:

                time.sleep(0.1)

    
    def run(self):
        
        found = False

        print(f"Searching for previous geneartions")
        population = sorted(glob.glob(f"{self.population_dir}/*/*/alive.txt"))
        
        if len(population) > 0:    
            
            if len(population) == self.population_size or len(population) == self.population_size*(self.number_of_child+1): 

                found = True
                
                for individual in population:

                    self.gen         = max(self.gen,         int(individual.split('/')[1]))
                    self.ind_counter = max(self.ind_counter, int(individual.split('/')[2]))
                
                if len(population) == self.population_size*(self.number_of_child+1): self.evolve_ras()

                print(f"Found Last Generation {self.gen} with last individual {self.ind_counter}")

            else: raise Exception(f"Corrupted Files, please delete the files in {self.population_dir}. Maybe the files were in between the evolution")

        else: found = False

        if found == False: self.create_initial_population()

        self.evolve()

    
    def create_initial_population(self):
        
        print(f"Did Not Found Any Last Generation\n")

        self.gen += 1

        generation_directory = f"{self.population_dir}/{self.gen}/"
        if not os.path.exists(generation_directory): os.makedirs(generation_directory)

        self.jobs = {}
        
        for _ in range(self.population_size): 
            self.ind_counter   += 1
                
            individual = f"{self.population_dir}/{self.gen}/{self.ind_counter}"
            if not os.path.exists(individual): os.makedirs(individual)

            self.create_lineage(individual)
            self.parallel_run(0, [0, individual, self.population, self.store_individual, self.gen])
        
        for deviceID in self.jobs:
            self.jobs[deviceID].join()
            if self.jobs[deviceID].exception is not None: raise Exception(f"{self.jobs[deviceID].exception[0]}, {self.jobs[deviceID].exception[1]}")
             
        self.population.write_log(f"\n")                          
        self.population.save_populations(f"{self.population_dir}/{self.gen}")

    
    def evolve_ras(self):

        population = sorted(glob.glob(f"{self.population_dir}/*/*/alive.txt"))
        assert len(population) == self.population_size

        self.population.read_populations(f"{self.population_dir}/{self.gen}")

        self.gen += 1

        generation_directory = f"{self.population_dir}/{self.gen}/"
        if not os.path.exists(generation_directory): os.makedirs(generation_directory)

        cluster_population = {'individual':[], 'dna':[], 'metrics':[], 'spectrum': []}
    
        for i, individual in enumerate(population):

            individual = individual.split('/alive.txt')[0]
            dna, metrics = self.read_individual(individual)

            cluster_population['individual']  += [individual]
            cluster_population['dna']         += [dna]
            cluster_population['metrics']     += [metrics]
            cluster_population['spectrum']    += [dna['graph'].graph['spectrum']]

        child_individuals = []
        
        self.jobs = {}
            
        for parent_individual in population * self.number_of_child: 

            self.ind_counter += 1

            parent_individual = parent_individual.split('/alive.txt')[0]                    
            parent_dna, parent_metrics = self.read_individual(parent_individual)

            child_individual = f"{self.population_dir}/{self.gen}/{self.ind_counter}"
            if not os.path.exists(child_individual): os.makedirs(child_individual)

            child_individuals += [child_individual]

            self.store_lineage(child_individual, parent_individual)
            self.parallel_run(1, [0, parent_dna, self.num_mutations, child_individual, self.population, self.store_individual, self.gen])

        for deviceID in self.jobs:

            self.jobs[deviceID].join()
            if self.jobs[deviceID].exception is not None: raise Exception(f"{self.jobs[deviceID].exception[0]}, {self.jobs[deviceID].exception[1]}")
                    
        for child_individual in child_individuals: 

            child_dna, child_metrics = self.read_individual(child_individual)
            child_spectrum           = child_dna['graph'].graph['spectrum']

            normalisation_spectrum = normalise(cluster_population['spectrum'] + [child_spectrum])
            distance               = [np.linalg.norm(x-normalisation_spectrum[-1]) for x in normalisation_spectrum[:-1]]
            closest_cluster_index  = distance.index(min(distance))

            if cluster_population['metrics'][closest_cluster_index]['fitness'] < child_metrics['fitness']:
                
                self.population.write_log(f"--> Worker changed  cluster {closest_cluster_index} head {cluster_population['individual'][closest_cluster_index].split('/')[2]} of fitness {cluster_population['metrics'][closest_cluster_index]['fitness']:.2f} to   {child_individual.split('/')[2]} of fitness {child_metrics['fitness']:.2f}\n")

                dead_individual_dir = cluster_population['individual'][closest_cluster_index]

                os.remove(f"{dead_individual_dir}/alive.txt")
                open(f"{dead_individual_dir}/dead_{self.gen}.txt", 'w').close()

                cluster_population['individual'][closest_cluster_index] = [child_individual]
                cluster_population['dna'][closest_cluster_index]        = [child_dna]
                cluster_population['metrics'][closest_cluster_index]    = [child_metrics]
                cluster_population['spectrum'] [closest_cluster_index]  = [child_spectrum]

            else:
                
                self.population.write_log(f"--> Worker retained cluster {closest_cluster_index} head {cluster_population['individual'][closest_cluster_index].split('/')[2]} of fitness {cluster_population['metrics'][closest_cluster_index]['fitness']:.2f} over {child_individual.split('/')[2]} of fitness {child_metrics['fitness']:.2f}\n")    
                
                dead_individual_dir = child_individual

                os.remove(f"{dead_individual_dir}/alive.txt")
                open(f"{dead_individual_dir}/dead_{self.gen}.txt", 'w').close()

        self.population.write_log(f"\n")    
        self.population.clean_populations()

        for pop in sorted(glob.glob(f"{self.population_dir}/*/*/alive.txt")):

            dna, metrics = self.read_individual(pop.split('/alive.txt')[0])
            self.population.update_populations(dna)

        self.population.save_populations(f"{self.population_dir}/{self.gen}")
                  

    def evolve(self):
        while True:
            self.evolve_ras()        

    
    def read_individual(self, individual):
        with open(f"{individual}/dna.pkl", 'rb')      as dna_file:      dna      = pickle.load(dna_file)
        with open(f"{individual}/metrics.pkl", 'rb')  as metrics_file:  metrics  = pickle.load(metrics_file)
        return dna, metrics

    
    def store_individual(self, individual, dna, metrics, mutations=None): 
        with open(f"{individual}/dna.pkl", 'wb')      as dna_file:      pickle.dump(dna, dna_file, pickle.HIGHEST_PROTOCOL)
        with open(f"{individual}/metrics.pkl", 'wb')  as metrics_file:  pickle.dump(metrics, metrics_file, pickle.HIGHEST_PROTOCOL)
        if mutations is not None: 
            with open(f"{individual}/mutations.pkl", 'wb') as mutations_file: pickle.dump(mutations, mutations_file, pickle.HIGHEST_PROTOCOL)

    
    def store_lineage(self, child_individual, parent_individual): 
        with open(f"{parent_individual}/lineage.pkl", 'rb') as lineage_file: lineage = pickle.load(lineage_file)
        lineage += [parent_individual.split('/')[2]]
        with open(f"{child_individual}/lineage.pkl", 'wb')  as lineage_file:  pickle.dump(lineage, lineage_file, pickle.HIGHEST_PROTOCOL)

    
    def create_lineage(self, individual): 
        with open(f"{individual}/lineage.pkl", 'wb')  as lineage_file:  pickle.dump([individual.split('/')[2]], lineage_file, pickle.HIGHEST_PROTOCOL)

                
def to_bool(arg_bool):

    if   arg_bool in ('True',  'true',  'T', 't', '1', 'Y', 'y'): return True
    elif arg_bool in ('False', 'false', 'F', 'f', '0', 'N', 'n'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Robust Architecture Search'))

    # Use for creating and mutating individual of population
    parser.add_argument("--use_cycles",             "-cy",   action="store_false",            help="if not to use cycles in populations")
    parser.add_argument("--use_adaptive_mutations", "-am",   action="store_true",             help="if use adaptive mutations")
    parser.add_argument("--use_random_params",      "-rp",   action="store_true",             help="if use random params")
    parser.add_argument("--use_non_squares",        "-sq",   action="store_true",             help="if use non-square kernels and strides")
    
    # Use for training an individual
    parser.add_argument("--use_augmentation",       "-au",   action="store_true",             help="if use augmented training")
    parser.add_argument("--use_limited_data",       "-ld",   action="store_true",             help="if use limited data for training")
    parser.add_argument("--dataset",                "-d",    default=2,            type=int,  help="Dataset to be used for training")
    parser.add_argument("--epochs",                 "-e",    default=50,           type=int,  help="Number of epochs to be used for a single individual")
    
    # Use for type of evolving generations
    parser.add_argument("--num_mutations",          "-m",    default=5,            type=int,  help="Number of mutations an individual undergoes")
    parser.add_argument("--num_mutated_child",      "-n",    default=2,            type=int,  help="Number of mutated individuals for a single parent")
    parser.add_argument("--population_size",        "-p",    default=25,           type=int,  help="Number of individuals in the population")
    
    # Use for changing directories
    parser.add_argument("--population_dir",         "-dir",  default="population", type=str,  help="Directory for storing all individuals")
    parser.add_argument("--log_dir",                "-log",  default="logs",       type=str,  help="Directory for logs")
    
    # Use for dry run
    parser.add_argument("--test",                   "-test", action="store_false",            help="Option for Dry Run to test, if the code runs. Please also check ./-log_dir-/exceptions.log too.")
    
    args = parser.parse_args()
    
    print(args)
    
    NeuroEvolution(args).run()
    
