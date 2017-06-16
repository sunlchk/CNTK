from __future__ import print_function
from __future__ import division

import cntk
import cntk.ops
import cntk.io
import cntk.train

from cntk.layers import Dense, Sequential
from cntk.io import StreamDef, StreamDefs, MinibatchSource, CTFDeserializer
from cntk.logging import ProgressPrinter

# Let's prepare data in the CTF format. It exactly matches
# the table above
INPUT_DATA = r'''|xy 0 0	|r 0
|xy 1 0	|r 1
|xy 0 1	|r 1
|xy 1 1	|r 0
'''

# Write the data to a temporary file
input_file = 'input'
with open(input_file, 'w') as f:
    f.write(INPUT_DATA)

# Create a network
xy = cntk.input_variable(2)
label = cntk.input_variable(1)

model = Sequential([
    Dense(2, activation=cntk.ops.tanh),
    Dense(1)])

z = model(xy)
loss = cntk.squared_error(z, label)

# Define our input data streams
streams = StreamDefs(
    xy = StreamDef(field='xy', shape=2),
    r = StreamDef(field='r', shape=1))

# Create a learner and a trainer and a progress writer to 
# output current progress
learner = cntk.sgd(model.parameters, cntk.learning_rate_schedule(0.1, cntk.UnitType.sample))
trainer = cntk.train.Trainer(z, (loss, loss), learner, ProgressPrinter(freq=10))

# Now let's create a minibatch source for out input file
mb_source = MinibatchSource(CTFDeserializer(input_file, streams))
input_map = { xy : mb_source['xy'], label : mb_source['r'] }

# Run a manual training minibatch loop
minibatch_size = 4
max_samples = 800
train = True
while train and trainer.total_number_of_samples_seen < max_samples:
    data = mb_source.next_minibatch(minibatch_size, input_map)
    train = trainer.train_minibatch(data)

# Run a manual evaluation loop ussing the same data file for evaluation
test_mb_source = MinibatchSource(CTFDeserializer(input_file, streams), randomize=False, max_samples=100)
test_input_map = { xy : test_mb_source['xy'], label : test_mb_source['r'] }
total_samples = 0
error = 0.
data = test_mb_source.next_minibatch(32, input_map)
while data:
    total_samples += data[label].number_of_samples 
    error += trainer.test_minibatch(data) * data[label].number_of_samples
    data = test_mb_source.next_minibatch(32, test_input_map)

print("Error %f" % (error / total_samples))

###################################################################################################################
###################################################################################################################
###################################################################################################################

# Run a manual training minibatch loop with checkpointing
import os

# Initialize main objects
mb_source = MinibatchSource(CTFDeserializer(input_file, streams))
input_map = { xy : mb_source['xy'], label : mb_source['r'] }

learner = cntk.sgd(model.parameters, cntk.learning_rate_schedule(0.1, cntk.UnitType.sample))
trainer = cntk.train.Trainer(z, (loss, loss), learner, ProgressPrinter(freq=10))

# Try to restore if the checkpoint exists
checkpoint = 'manual_loop_checkpointed'

if os.path.exists(checkpoint):
    print("Trying to restore from checkpoint")
    mb_source_state = trainer.restore_from_checkpoint(checkpoint)
    mb_source.restore_from_checkpoint(mb_source_state)
    print("Restore has finished successfully")
else:
    print("No restore file found")
    
checkpoint_frequency = 100
last_checkpoint = 0
train = True
while train and trainer.total_number_of_samples_seen < max_samples:
    data = mb_source.next_minibatch(minibatch_size, input_map)
    train = trainer.train_minibatch(data)
    if trainer.total_number_of_samples_seen / checkpoint_frequency != last_checkpoint:
        mb_source_state = mb_source.get_checkpoint_state()
        trainer.save_checkpoint(checkpoint, mb_source_state)
        last_checkpoint = trainer.total_number_of_samples_seen / checkpoint_frequency


###################################################################################################################
###################################################################################################################
###################################################################################################################

# Run a manual training minibatch loop with distributed learner
checkpoint = 'manual_loop_distributed'

mb_source = MinibatchSource(CTFDeserializer(input_file, streams))
input_map = { xy : mb_source['xy'], label : mb_source['r'] }

# Make sure the learner is distributed
learner = cntk.distributed.data_parallel_distributed_learner(cntk.sgd(model.parameters, cntk.learning_rate_schedule(0.1, cntk.UnitType.sample)))
trainer = cntk.train.Trainer(z, (loss, loss), learner, ProgressPrinter(freq=10))

if os.path.exists(checkpoint):
    print("Trying to restore from checkpoint")
    mb_source_state = trainer.restore_from_checkpoint(checkpoint)
    mb_source.restore_from_checkpoint(mb_source_state)
else:
    print("No restore file found")

last_checkpoint = 0
train = True
partition = cntk.distributed.Communicator.rank()
num_partitions = cntk.distributed.Communicator.num_workers()
while train:
    data = {}
    if trainer.total_number_of_samples_seen < max_samples:
        # Make sure each worker gets its own data only
        data = mb_source.next_minibatch(minibatch_size_in_samples = minibatch_size,
                                        input_map = input_map, device = cntk.use_default_device(), 
                                        num_data_partitions=num_partitions, partition_index=partition)
    train = trainer.train_minibatch(data)
    if trainer.total_number_of_samples_seen / checkpoint_frequency != last_checkpoint:
        mb_source_state = mb_source.get_checkpoint_state()
        trainer.save_checkpoint(checkpoint, mb_source_state)
        last_checkpoint = trainer.total_number_of_samples_seen / checkpoint_frequency

# When you use distributed learners, please call finalize MPI at the end of your script, 
# see the next cell.
# cntk.distributed.Communicator.finalize()


###################################################################################################################
###################################################################################################################
###################################################################################################################


checkpoint = 'training_session'

# Minibatch sources
mb_source = MinibatchSource(CTFDeserializer(input_file, streams))
test_mb_source = MinibatchSource(CTFDeserializer(input_file, streams), randomize=False, max_samples=100)

learner = cntk.distributed.data_parallel_distributed_learner(cntk.sgd(model.parameters, cntk.learning_rate_schedule(0.1, cntk.UnitType.sample)))
trainer = cntk.train.Trainer(z, (loss, loss), learner, ProgressPrinter(freq=1))

test_config=cntk.TestConfig(minibatch_source = test_mb_source,
                            model_inputs_to_streams={ xy : test_mb_source['xy'], label : test_mb_source['r'] })

session = cntk.training_session(
    trainer = trainer, mb_source = mb_source, 
    mb_size = minibatch_size, 
    model_inputs_to_streams={ xy : mb_source['xy'], label : mb_source['r'] },
    max_samples = max_samples,
    checkpoint_config=cntk.CheckpointConfig(frequency=checkpoint_frequency, filename=checkpoint),
    test_config=cntk.TestConfig(minibatch_source = test_mb_source, minibatch_size = minibatch_size,
                                model_inputs_to_streams={ xy : test_mb_source['xy'], label : test_mb_source['r'] }))

session.train()

# When you use distributed learners, please call finalize MPI at the end of your script
cntk.distributed.Communicator.finalize()

