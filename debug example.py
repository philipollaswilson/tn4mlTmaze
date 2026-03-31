import numpy as np

def one_hot(val, size):
    vec = [0] * size
    vec[val] = 1
    return vec
              
# Create dataset 
X = np.empty((1, 4), dtype=int)
number = 1
for context in range(2,4):
    for a1 in range(3):
        
        # If mouse goes to cue it sees Right or Left
        if a1 == 2:
            o2 = context
            
        # If it goes Left  or Right it sees Cheese or Shock
        elif a1 == 1:
            o2 = 3 - context
            
        else:
            o2 = context - 2
            
        for a2 in range(3):
            if a1 != 2: # TRAP
                o3 = o2
            else:
                if a2 == 2:
                    o3 = context
            
                # If it goes Left  or Right it sees Cheese or Shock
                elif a2 == 1:
                    o3 = 3 - context
                    
                else:
                    o3 = context - 2
                    
            # print(f"number{number} a1:{a1} o2:{o2} a2:{a2} o3:{o3}")
            number += 1
            datapoint = np.array([a1, o2, a2, o3])

            X = np.vstack((X, datapoint))

  
X = X[1:] # Remove the first empty row          
# 18 possible paths and 4 variables (a1, o2, a2, o3)


from tn4ml.models.mps import MatrixProductState
from tn4ml.metrics import NegLogLikelihood
import numpy as np
import optax
from tn4ml.util import EarlyStopping 
import jax.numpy as jnp
from tn4ml.embeddings import OneHotEmbedding

# Initialize MPS
max_bond_dim = 16

import jax.numpy as jnp

def identity_3d_cubical(shape, backend='numpy'):
    """
    Creates a 3D identity-like tensor where only elements with i == j == k are 1.

    Args:
        shape (tuple): Shape of the 3D tensor (e.g., (3, 3, 3)).
        backend (str): 'numpy' or 'jax'. Defaults to 'numpy'.

    Returns:
        ndarray: 3D identity-like tensor.

    Example:
        >>> identity_3d_cubical((3, 3, 3))
        array([[[1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
               [[0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]],
               [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]]])
    """
    if backend == 'jax':
        xp = jnp
    else:
        xp = np

    def _identity_rule(i, j, k):
        return xp.where(i == j == k, 1, 0)

    return xp.fromfunction(
        xp.vectorize(_identity_rule, otypes=[float]),
        shape,
        dtype=float
    )

A1 = identity_3d_cubical((max_bond_dim, max_bond_dim, 4)) # Should be 3, but in order to pass one Embed function, actions will have a 4th null state.
O2 = identity_3d_cubical((max_bond_dim, max_bond_dim, 4))
A2 = identity_3d_cubical((max_bond_dim, max_bond_dim, 4)) # Idem
O3 = identity_3d_cubical((max_bond_dim, max_bond_dim, 4))
tensors = [A1, O2, A2, O3]

model = MatrixProductState(tensors)

# define training parameters
epochs = 100
batch_size = 18
optimizer = optax.adam
strategy = 'global'
loss = NegLogLikelihood
train_type = 0 # TrainingType.UNSUPERVISED

embedding = OneHotEmbedding(num_states=4)
learning_rate = 1e-4
earlystop = EarlyStopping(min_delta=0, patience=10, monitor='loss', mode='min')
device = 'cpu'

model.configure(optimizer=optimizer, strategy=strategy, loss=loss, train_type=train_type, learning_rate=learning_rate, device=device)

history = model.train(X,
            epochs=epochs,
            batch_size=batch_size,
            embedding = embedding,
            normalize = True,
            dtype = jnp.float64,
            earlystop = earlystop,
            cache = True,
            )