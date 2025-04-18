import numpy as np
import scipy.stats
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import SGD
import multiprocessing
from multiprocessing import shared_memory
import random
import math
import time
import ast

# This script is used to apply the MIA attack for the experiments with real data given in section 4.2 of the paper 
# "MCMC for Bayesian estimation of Differential Privacy from Membership Inference Attacks"

def LearnAndDecide(l_star, alpha, l_0, l_1):
    """Learns the hypotheses H_0 and H_1 and returns a decision.

    Returns:
        b: Decision value.
    """

    # Fit normal distributions for H_0 and H_1
    mu_l_0, var_l_0 = FitNormal(l_0)
    mu_l_1, var_l_1 = FitNormal(l_1)

    # Decision
    b_l = LRTNormal(l_star, mu_l_0, var_l_0, mu_l_1, var_l_1, alpha)

    if b_l == -1:
        return -1
    return b_l

def FitNormal(x):
    return np.mean(x), np.var(x, ddof=1)      

def LRTNormal(y, mu_0, var_0, mu_1, var_1, alpha):
    """Applies the likelihood ratio test and returns a decision

    Args:
        y: Expected loss.
        mu_0: Mean of sample loss values over H_0.
        var_0: Variance of sample loss values over H_0.
        mu_1: Mean of sample loss values over H_1.
        var_1: Variance of sample loss values over H_1.
        alpha: Target false positive (FP) rate.

    Returns:
        b: Decision value.
    """
    try:
        A = (mu_0/var_0 - mu_1/var_1) / (1/var_1 - 1/var_0)
        delta = mu_0/math.sqrt(var_0) + (1/math.sqrt(var_0)) * A
    except:
        print("Error: Float div. by 0.")
        return -1
    
    # print(scipy.stats.ncx2.ppf(alpha, 1, delta**2))
    if (pow((y + A), 2) <= (var_0 * scipy.stats.ncx2.ppf(alpha, 1, pow(delta,2)))) and (var_0 > var_1):
        b = 1
    elif (pow((y + A), 2) >= (var_0 * scipy.stats.ncx2.ppf(1-alpha, 1, pow(delta,2)))) and (var_0 < var_1):
        b = 1
    else:
        b = 0
    return b


def EvalModels(j, x_x, x_y, shm_name_x_0, shm_name_y_0, shape_x_0, shape_y_0, shm_name_x_1, shm_name_y_1, shape_x_1, shape_y_1, dtype, BATCH_SIZE, EPOCHS, DP_STD):
    # *! Set all random seeds to None to ensure randomness using multiprocessing
    tf.random.set_seed(None) 
    random.seed(None)
    np.random.seed(None)

    # Attach to existing shared memory
    existing_shm_x_0 = shared_memory.SharedMemory(name=shm_name_x_0)
    existing_shm_y_0 = shared_memory.SharedMemory(name=shm_name_y_0)
    existing_shm_x_1 = shared_memory.SharedMemory(name=shm_name_x_1)
    existing_shm_y_1 = shared_memory.SharedMemory(name=shm_name_y_1)

    # Reconstruct shared arrays
    D_0_x = np.ndarray(shape_x_0, dtype=dtype, buffer=existing_shm_x_0.buf)
    D_0_y = np.ndarray(shape_y_0, dtype=dtype, buffer=existing_shm_y_0.buf)
    D_1_x = np.ndarray(shape_x_1, dtype=dtype, buffer=existing_shm_x_1.buf)
    D_1_y = np.ndarray(shape_y_1, dtype=dtype, buffer=existing_shm_y_1.buf)

    # * Enable for same initial weight experiments.
    keras.utils.set_random_seed(0)
    tf.random.set_seed(0) 
    random.seed(0)
    np.random.seed(0)

    # Generate the model : theta_0        
    curr_model_0 = Sequential()
    curr_model_0.add(Dense(128, activation="relu", input_shape=(28 * 28,)))
    curr_model_0.add(Dense(10, activation="softmax"))

    curr_model_0.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # * Enable for same initial weight experiments.
    tf.random.set_seed(None)  # Disables deterministic behavior (for DP and epoch-wise shuffling)
    random.seed(None)
    np.random.seed(None)

    curr_model_0.fit(
        D_0_x,
        D_0_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        shuffle=True
    )

    # Apply DP noise (comment out the noise addition if DP needs to be disabled)
    weights_0 = curr_model_0.get_weights()
    steps = len(weights_0)
    weight_final = []

    for i in range(steps):
        if i % 2 == 0:     
            noise = np.random.normal(0, DP_STD, (len(weights_0[i]), len(weights_0[i][0])))
            weight_final.append(weights_0[i] + noise)
            # print(noise[0][:5])
        else:
            noise = np.random.normal(0, DP_STD, len(weights_0[i]))
            weight_final.append(weights_0[i] + noise)
            # print(noise[:5])

    curr_model_0.set_weights(weight_final)

    l_0 = curr_model_0.evaluate(x_x, x_y, verbose=0)[0]

    # * Enable for same initial weight experiments.
    keras.utils.set_random_seed(0)
    tf.random.set_seed(0) 
    random.seed(0)
    np.random.seed(0)

    # Generate the model : theta_1
    curr_model_1 = Sequential()
    curr_model_1.add(Dense(128, activation="relu", input_shape=(28 * 28,)))
    curr_model_1.add(Dense(10, activation="softmax"))

    curr_model_1.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # * Enable for same initial weight experiments.
    tf.random.set_seed(None)  # Disables deterministic behavior (for DP and epoch-wise shuffling)
    random.seed(None)
    np.random.seed(None)
    
    curr_model_1.fit(
        D_1_x,
        D_1_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        shuffle=True
    )

    # Apply DP noise (comment out the noise addition if DP needs to be disabled)
    weights_1 = curr_model_1.get_weights()
    steps = len(weights_1)
    weight_final = []

    for i in range(steps):
        if i % 2 == 0:
            noise = np.random.normal(0, DP_STD, (len(weights_0[i]), len(weights_0[i][0])))
            weight_final.append(weights_1[i] + noise)
            # print(noise[0][:5])
        else:
            noise = np.random.normal(0, DP_STD, len(weights_1[i]))
            weight_final.append(weights_1[i] + noise)
            # print(noise[:5])

    curr_model_1.set_weights(weight_final)

    l_1 = curr_model_1.evaluate(x_x, x_y, verbose=0)[0]
    
    return l_0, l_1


def IterateAlphas(N, l_0, l_1, alpha, file_name):
    X = 0
    Y = 0
    for j in range(N):
        # print(f"{j}...")

        b_ = LearnAndDecide(l_0[j], alpha, l_0[:j]+l_0[j+1:], l_1)
        if b_ == 1:
            X += 1

        b = LearnAndDecide(l_1[j], alpha, l_0, l_1[:j]+l_1[j+1:])
        if b == 0:
            Y += 1 

        if b == -1 or b_ == -1:
            return -1, -1
        
    return X, Y


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn") # Spawn is default in MacOS, fork is default in Linux. Fork can be problematic but is faster.
    NO_OF_EXPR = 20 # (n)
    for expr_no in range(NO_OF_EXPR):
        SUBSET_SIZE = 1000

        NUM_ROWS = NUM_COLS = 28
        NUM_CLASSES = 10
        BATCH_SIZE = 100
        EPOCHS = 200

        DP_STD = 0.1
        N = 100

        file_name = f'dp_attack_w_DP_fixed_weights_{DP_STD}.txt'

        with open(file_name, 'a') as file:
            file.write("\n")
            file.write(f"* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n")
            file.write("\n")

        ############################################################
        ## Load the dataset
        ############################################################

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Reshape data
        x_train = x_train.reshape((x_train.shape[0], NUM_ROWS * NUM_COLS))
        x_train = x_train.astype("float32") / 255
        x_test = x_test.reshape((x_test.shape[0], NUM_ROWS * NUM_COLS))
        x_test = x_test.astype("float32") / 255

        # Categorically encode labels
        y_train = to_categorical(y_train, NUM_CLASSES)
        y_test = to_categorical(y_test, NUM_CLASSES)

        # # * For reading a (D, x) pair
        # Read the D and x indices from the saved file
        with open("sample_info_n_1000.txt", 'r') as file:
            saved_data = file.read().split("\n")
        
        indices = ast.literal_eval(saved_data[2*expr_no])  
        ind_x = int(saved_data[2*expr_no+1])  
        print("The index of x:", ind_x)

        x_x = x_train[indices[ind_x]] # We use the index of x in the x_train dataset
        x_y = y_train[indices[ind_x]]

        x_x = x_x.reshape(1, 784)
        x_y = x_y.reshape(1, 10)

        # ! DO NOT UNCOMMENT : for generating a new (D, x) pair
        # indices = random.sample(range(len(x_train)), SUBSET_SIZE)
        # ind_x = random.randint(0, len(indices)) # Index of x in the "indices" array

        # x_x = x_train[indices[ind_x]] # We use the index of x in the x_train dataset
        # x_y = y_train[indices[ind_x]]

        # x_x = x_x.reshape(1, 784)
        # x_y = x_y.reshape(1, 10)

        # with open("sample_info_n_1000.txt", 'a') as file:
        #     file.write(str(indices))
        #     file.write("\n")
        #     file.write(str(ind_x))
        #     file.write("\n")

        ############################################################

        indices = np.array(indices)

        # Take the subset using np indexing
        D_1_x = x_train[indices]
        D_1_y = y_train[indices]

        D_0_x = np.delete(D_1_x, ind_x, axis=0)  # Remove row based on the "indices" index, which is equivalent to D's indices
        D_0_y = np.delete(D_1_y, ind_x, axis=0)  # Remove corresponding label

        # Save header
        with open(file_name, 'a') as file:
            file.write(f"|D| : {SUBSET_SIZE}, |D_0| : {len(D_0_x)}, |D_1| : {len(D_1_x)}\n")
            file.write(f"N = N_0 = N_1 : {N}\n")
            file.write(f"Epochs : {EPOCHS}, Batch size : {BATCH_SIZE}\n")
            file.write(f"DP STD : {DP_STD}\n")

        models_0 = []
        models_1 = []

        l_0 = [0] * N
        l_1 = [0] * N

        # Create shared memory for the datasets (D)
        shm_x_0 = shared_memory.SharedMemory(create=True, size=D_0_x.nbytes)
        shm_y_0 = shared_memory.SharedMemory(create=True, size=D_0_y.nbytes)

        shm_x_1 = shared_memory.SharedMemory(create=True, size=D_1_x.nbytes)
        shm_y_1 = shared_memory.SharedMemory(create=True, size=D_1_y.nbytes)

        # Copy data to shared memory
        shared_x_0 = np.ndarray(D_0_x.shape, dtype=D_0_x.dtype, buffer=shm_x_0.buf)
        shared_y_0 = np.ndarray(D_0_y.shape, dtype=D_0_y.dtype, buffer=shm_y_0.buf)
        np.copyto(shared_x_0, D_0_x)
        np.copyto(shared_y_0, D_0_y)

        shared_x_1 = np.ndarray(D_1_x.shape, dtype=D_1_x.dtype, buffer=shm_x_1.buf)
        shared_y_1 = np.ndarray(D_1_y.shape, dtype=D_1_y.dtype, buffer=shm_y_1.buf)
        np.copyto(shared_x_1, D_1_x)
        np.copyto(shared_y_1, D_1_y)

        task_data = [(j, x_x, x_y, shm_x_0.name, shm_y_0.name, D_0_x.shape, D_0_y.shape, shm_x_1.name, shm_y_1.name, D_1_x.shape, D_1_y.shape, D_0_x.dtype, BATCH_SIZE, EPOCHS, DP_STD) for j in range(N)]

        start = time.time()

        with multiprocessing.Pool() as executor:
            results = list(executor.starmap(EvalModels, task_data))
        
        executor.close()
        executor.join()

        end = time.time()
        length = end - start

        print("Model generation took", length, "seconds!")

        l_0, l_1 = zip(*results)

        # Cleanup shared memory
        shm_x_0.close()
        shm_x_0.unlink()
        shm_y_0.close()
        shm_y_0.unlink()

        shm_x_1.close()
        shm_x_1.unlink()
        shm_y_1.close()
        shm_y_1.unlink()

        # Save the (ell, nu) values
        with open(file_name, 'a') as file:
            file.write(f"l_0 : {str(l_0)}\n")
            file.write(f"l_1 : {str(l_1)}\n")

        task_data = [(N, l_0, l_1, alpha, file_name) for alpha in [a/10 for a  in range(0, 11)]]

        start = time.time()

        with multiprocessing.Pool(11) as executor:
            results = list(executor.starmap(IterateAlphas, task_data))
        
        executor.close()
        executor.join()

        end = time.time()
        length = end - start

        print("Decision calculations took", length, "seconds!")

        X, Y = zip(*results)

        if X == -1 or Y == -1:
            break

        # with open(file_name, 'a') as file:
        #     i = 0
        #     for alpha in [a/10 for a  in range(0, 11)]:
        #         file.write(f"{alpha}: ({X[i]}, {Y[i]}),\n"); i+= 1
        #     file.write("\n")

        print(X,Y)    
