# Split RGCN architecture

## Description

The proposed architecture is the one showing in the following picture

## Results

Some initial results from the proposed architecture according to the selected KG embedding model. The `Rel Emb Dim` and `Epochs` refer to the training of the KG embeddings, not the GCN, as well as the `Hits @ 10` metric. `Accuracy` is the classification accuracy achieved by the GCN model eventually.

All models but `Baseline HGT` were trained for 30 epochs, while `Baseline HGT` got trained for 100.

Model | Rel Emb Dim | Ent Emb Dim | Epochs | Hits @ 10 | Best Val Accuracy 
--- | --- | --- | --- |--- |--- 
TransR | 30 | 128 | 25 | 0.27353 | 0.6
TransR | 30 | 128 | 50 | 0.70721 | 0.59
TransH | 30 | 128 | 25 | 0.0116 | 0.6
TransH | 30 | 128 | 50 | 0.01345 | 0.585
RotatE | 30 | 128 | 25 | 1.0 | 0.5775
RotatE | 30 | 128 | 50 | 1.0 | 0.5875
DistMult | 30 | 128 | 25 | 0.25088 | 0.6075
DistMult | 30 | 128 | 50 | 0.2512 | 0.6125
ComplEx | 30 | 128 | 25 | 0.00094 | 0.575
ComplEx | 30 | 128 | 50 | 0.001 | 0.57
Look-Up | 2 | - | - | - | 0.595
None | - | - | - | - | 0.59
Baseline HGT | - | - | - | - | 0.615