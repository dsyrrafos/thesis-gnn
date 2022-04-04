from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# create paths to data
# train_tab.txt and test_tab.txt are simply tab separated triples in the form
# 'head\trelation\ttail'

train_path = "./data/IMDB/processed/data.pt"
test_path = "./data/IMDB/processed/data.pt"

# Load training data
training = TriplesFactory.from_path(
    train_path,
    create_inverse_triples=True
)

# Split training to train and valid
training, validation = training.split([0.9, 0.1], random_state=42)

# Load test data using the entity_to_id, relation_to_id from the previous loader
testing = TriplesFactory.from_path(
    test_path,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
    create_inverse_triples=True
)

#
result = pipeline(

    training=training,

    testing=testing,
    #validation=validation, # If yoy want to validate results
    model='TransR',
    random_seed=42,
    model_kwargs={"embedding_dim":200},
    training_kwargs={"num_epochs":25},
    #stopper='early', # early stopping arguments. You need the validation set with this.
    #stopper_kwargs=dict(frequency=3, patience=3, relative_delta=0.002),
    # epochs=5,  # short epochs for testing - you should go higher
)
# Save mode to a directory. Yoy can load it afterwards
result.save_to_directory('Models_pykeen/Rotate_25')

