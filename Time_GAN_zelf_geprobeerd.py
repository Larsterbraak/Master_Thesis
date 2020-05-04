"""
Python script that will do: Perform the Time_GAN coded in Tensorflow 2.1.0
Created on Fri May  1 10:29:49 2020
@author: G.T.F. (Lars) ter Braak

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import os
import tensorflow as tf

# Change the working directory and import the data
os.chdir('C:/Users/s157148/Documents/Research')
df = pd.read_csv('Data/df_full_pre_ester.csv', sep = ';')

# Show the 25th, 75th and weighted average of the per-ESTER
plt.figure()
plt.plot(df.R25, color='C0', label = '25-percentile')
plt.plot(df.WT, color = 'C1', label = 'ESTER')
plt.plot(df.R75, color='C2', label = '75-percentile')
plt.title('ESTER over pre-ester period')
plt.xlabel('Days')
plt.ylabel('Interest rate')
plt.legend()
plt.show()

def create_dataset(df, seq_length):
    dataX = []
    
    # Make lookback periods of seq_length in the data
    for i in range(0, len(df) - seq_length):
        _df = df[i : i + seq_length]
        dataX.append(_df)
    
    # Create random permutations to make it more i.i.d.
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
        
    return outputX

# Convert the interest rates to daily differences
df.iloc[1:,[1,2,4]] = np.diff(df[['R25', 'R75', 'WT']], axis =0)
df = df.iloc[1:]

from sklearn.preprocessing import MinMaxScaler
df = MinMaxScaler().fit_transform(df)

# Create the dataset and reshape in the correct format
df = create_dataset(df, 20)
df = np.reshape(df, newshape=(628,20,5))

# Split the data in train and test
X_train = df[0:400,:,:]
X_test = df[400:,:,:]

# Clear up memory space
del df

# Change all layers to have dtype float64
tf.keras.backend.set_floatx('float64')

# Make the data into tensorflow data
X_train = tf.data.Dataset.from_tensor_slices((X_train))
X_train = X_train.batch(25)
X_test = tf.data.Dataset.from_tensor_slices((X_test))
X_test = X_test.batch(25)

# =============================================================================
# Make the customized models
# =============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Model
tf.keras.backend.clear_session()

# Embedder network in Tensorflow 2.x
class Embedder(Model):
  def __init__(self):
    super(Embedder, self).__init__()
    self.LSTM1 = LSTM(units=25, 
                      return_sequences=True,
                      input_shape=(20,5),
                      name = 'embedder/LSTM1')
    
    self.LSTM2 = LSTM(units=25,
                      return_sequences=True,
                      name = 'embedder/LSTM2')
    
    self.Dense1 = Dense(units=25,
                        activation='sigmoid',
                        name = 'embedder/Dense1')

  def call(self, x):
    x = self.LSTM1(x)
    x = self.LSTM2(x)
    return self.Dense1(x)
    
# Create an instance of the embedder model
embedder_model = Embedder()

# Recovery network in Tensorflow 2.x
class Recovery(Model):
  def __init__(self):
    super(Recovery, self).__init__()
    self.LSTM1 = LSTM(units=15, 
                      return_sequences=True, 
                      input_shape=(20,25), 
                      name = 'recovery/LSTM1')
    self.LSTM2 = LSTM(units=15, 
                      return_sequences=True,
                      name = 'recovery/LSTM2')
    self.Dense1 = Dense(units=5, 
                        activation='sigmoid', 
                        name = 'recovery/Dense1')

  def call(self, x):
    x = self.LSTM1(x)
    x = self.LSTM2(x)
    return self.Dense1(x)

# Create an instance of the recovery model
recovery_model = Recovery()

# Generator network in Tensorflow 2.x
class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.LSTM1 = LSTM(units=25, 
                      return_sequences=True, 
                      input_shape=(20,5), 
                      name = 'generator/LSTM1')
        self.LSTM2 = LSTM(units=25, 
                      return_sequences=True,
                      name = 'generator/LSTM2')
        self.Dense1 = Dense(units=25, 
                        activation='sigmoid', 
                        name = 'generator/Dense1')
    
    def call(self, x):
        x = self.LSTM1(x)
        x = self.LSTM2(x)
        return self.Dense1(x)

class Supervisor(Model):
    def __init__(self):
        super(Supervisor, self).__init__()
        self.LSTM1 


# Discriminator network for GAN in latent space in Tensorflow 2.x
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

# Random vector generation Z
def RandomGenerator(batch_size, z_dim):
    Z_minibatch = list()
    
    for i in range(batch_size): 
        Z_minibatch.append(np.random.uniform(0., 1, [z_dim[0], z_dim[1]]))

    return Z_minibatch

# =============================================================================
# Create the loss object, optimizer, and training function
# =============================================================================

# Create a loss object
loss_object = tf.keras.losses.MeanSquaredError()

# Create a Adam optimizer
optimizer = tf.keras.optimizers.Adam()

# Metrics to track during training
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_step_auto_encode(X_train):
  with tf.GradientTape() as tape:
    # Apply the Embedder to the data
    e_pred_train = embedder_model(X_train)
    #embedder_loss = loss_object(X, y)
    
    # Apply the Recovery to the predicted hidden states    
    r_pred_train = recovery_model(e_pred_train)
    
    # Compute the loss function for the LSTM autoencoder
    r_loss_train = loss_object(X_train, r_pred_train)
  
  # Compute the gradients with respect to the Embedder and Recovery vars
  gradients = tape.gradient(r_loss_train, 
                            embedder_model.trainable_variables \
                            + recovery_model.trainable_variables)
  
  # Apply the gradients to the Embedder and Recovery vars
  optimizer.apply_gradients(zip(gradients, 
                                embedder_model.trainable_variables +
                                recovery_model.trainable_variables))
  train_loss(r_loss_train)

@tf.function
def test_step_auto_encode(X_test):
    
    # Apply the Embedder to the data
    e_pred_test = embedder_model(X_test)
    #embedder_loss = loss_object(X, y)
    
    # Apply the Recovery to the predicted hidden states    
    r_pred_test = recovery_model(e_pred_test)
    
    # Compute the loss function for the LSTM autoencoder
    r_loss_test = loss_object(X_test, r_pred_test)
  
    test_loss(r_loss_test)
    
# =============================================================================
# Start with embedder training    
# =============================================================================
EPOCHS = 50

# Train the embedder for the input data
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    test_loss.reset_states()
    
    # Train over the complete train dataset
    for x_train in X_train:
        train_step_auto_encode(x_train)
    
    # Test over the complete test dataset
    for x_test in X_test:
        test_step_auto_encode(x_test)

    template = 'Epoch {}, Loss: {}, Test Loss: {}'
    print(template.format(epoch+1, 
                          train_loss.result(),
                          test_loss.result() ) )

print('Finished Embedding Network Training')

# =============================================================================
# Continu with supervised loss
# =============================================================================
# Create a loss object
loss_object = tf.keras.losses.MeanSquaredError()

# Create a Adam optimizer
optimizer = tf.keras.optimizers.Adam()

# Metrics to track during training
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_step_auto_encode(X_train):
  with tf.GradientTape() as tape:
    # Apply the Embedder to the data
    e_pred_train = embedder_model(X_train)
    #embedder_loss = loss_object(X, y)
    
    # Apply the Recovery to the predicted hidden states    
    r_pred_train = recovery_model(e_pred_train)
    
    # Compute the loss function for the LSTM autoencoder
    r_loss_train = loss_object(X_train, r_pred_train)
  
  # Compute the gradients with respect to the Embedder and Recovery vars
  gradients = tape.gradient(r_loss_train, 
                            embedder_model.trainable_variables \
                            + recovery_model.trainable_variables)
  
  # Apply the gradients to the Embedder and Recovery vars
  optimizer.apply_gradients(zip(gradients, 
                                embedder_model.trainable_variables +
                                recovery_model.trainable_variables))
  train_loss(r_loss_train)

# Train the supervised loss first such
# that the temporal relations are preserved
for epoch in range(EPOCHS):
    
    # Get a batch from the data
    
    # Make a random generation
    Z_mb = random_generator(batch_size, z_dim)

# =============================================================================
# Different shit
# =============================================================================

# Embedder network
def embedder(X):
    #with tf.variable_scope('embedder', reuse= tf.AUTO_REUSE):
    embedder = Sequential()
    embedder.add(LSTM(units=25, return_sequences=True, input_shape=(20,5), name = 'embedder/LSTM1'))
    embedder.add(LSTM(units=25, return_sequences=True, name = 'embedder/LSTM2'))
    embedder.add(Dense(25, activation='sigmoid', name = 'embedder/Dense1'))
        
    # Predict the temporal hidden latent space
    H = embedder.predict(X)
    return H


rnn_cells = [tf.keras.layers.LSTMCell(25) for _ in range(2)]
stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
lstm_layer = tf.keras.layers.RNN(stacked_lstm)
    
# Recovery network
def recovery(H):
    #with tf.variable_scope('recovery', reuse= tf.AUTO_REUSE):
    recovery = Sequential()
    recovery.add(LSTM(units=15, return_sequences=True, input_shape=(20,25), name = 'recovery/LSTM1'))
    recovery.add(LSTM(units=15, return_sequences=True, name = 'recovery/LSTM2'))
    recovery.add(Dense(5, activation='sigmoid', name = 'recovery/Dense1'))
        
    # Predict the recovery
    X_tilde = recovery.predict(H)
    return X_tilde

# Embedder Networks
H = embedder(df)
X_tilde = recovery(H)



# How to do it in TensorFlow 1.x
# Variables for the embedder network
e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]

# Loss for the embedder network
def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

# Loss for the embedder network
loss = tf.losses.mean_squared_error(df, X_tilde)
E_loss0 = 10 * tf.sqrt(loss)

from tensorflow.train import GradientDescentOptimizer

opt = GradientDescentOptimizer(0.1)
grads_and_vars = opt.compute_gradients(loss, e_vars)
opt.apply_gradients(grads_and_vars)

from tensorflow import train
# Define the optimizer and the list of variables it should update
E0_solver = train.AdamOptimizer().minimize(E_loss0, var_list= e_vars + r_vars)

# Start the tensorflow sessions
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# Start the embedding learning
print('Start Embedding Network Training')
for itt in range(10):
sess.run([E0_solver], feed_dict={df: df})        

# Build a RNN generator network
def generator(Z):
    generator = Sequential(name= 'generator')
    generator.add(LSTM(units=15, return_sequences=True, input_shape=(20, 5)))
    generator.add(LSTM(units=15, return_sequences=True))
    generator.add(Dense(10, activation='sigmoid'))
    
    # Predict the temporal hidden latent space
    E = generator.predict(Z)
    
    return E
    
    
    
    
# Tensorboard callback for keras
tensorboard_callback = keras.callbacks.TensorBoard('./graphs',
    histogram_freq=1, # how often to compute activation and weight histograms
    write_graph=True, # visualize the graph
    write_grads=True, # visual gradient histogram
    write_images=True, # visualize model weights as an image
    # embeddings_freq=1, # how often to visualize embeddings
    # embeddings_layer_names=['...'], # names of embedding layers to visualize; wouldn't work on this model
    update_freq='epoch' # update TensorBoard every epoch
)