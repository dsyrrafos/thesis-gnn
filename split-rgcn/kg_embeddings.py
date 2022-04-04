import os
import itertools
import gc

import numpy as np
import pandas as pd

import torch
import pykeen

print(pykeen.env())

from torch_geometric.datasets import IMDB

dataset = IMDB(root='./data/imdb')
print('Dataset:', dataset)
print('Number of graphs:', len(dataset))

data = dataset[0]
print(data)

num_classes = len(data['movie'].y.unique())
print('Number of classes:', num_classes)
print('Classes:', data['movie'].y.unique())

print('Ranges of each node type')
print('Movie:',0,'-',data['movie'].x.size()[0]-1)
print('Director:', data['movie'].x.size()[0], '-', 
        data['movie'].x.size()[0]+data['director'].x.size()[0]-1)
print('Actor:', data['movie'].x.size()[0]+data['director'].x.size()[0], 
        data['movie'].x.size()[0]+data['director'].x.size()[0] +
        data['actor'].x.size()[0]-1)

movie_size = data['movie'].x.size()[0]
director_size = data['director'].x.size()[0]
print(movie_size, director_size)
movie_size = data['movie'].x.size()[0]
director_size = data['director'].x.size()[0]
offset_director = torch.tensor([[0],[movie_size]])
offset_director = offset_director.tile(1, 
                data[('movie', 'to', 'director')].edge_index.size()[1])
movie_to_director = data[('movie', 'to', 'director')].edge_index + offset_director
offset_actor = torch.tensor([[0],[movie_size + director_size]])
offset_actor = offset_actor.tile(1, 
                data[('movie', 'to', 'actor')].edge_index.size()[1])
movie_to_actor = data[('movie', 'to', 'actor')].edge_index + offset_actor
print(movie_to_director.size(), movie_to_actor.size())

pad = torch.zeros(movie_to_actor.size()[1])
movie_to_actor = torch.column_stack((movie_to_actor[0],pad,movie_to_actor[1]))
pad = torch.ones(movie_to_director.size()[1])
movie_to_director = torch.column_stack((movie_to_director[0],pad,movie_to_director[1]))
print(movie_to_director.size(), movie_to_actor.size())

triples = torch.concat((movie_to_director, movie_to_actor))
triples.size()

print(triples.size())

entity_ids = [i for i in range (data.num_nodes)]
relation_ids = [0,1]

print(triples.long().type())