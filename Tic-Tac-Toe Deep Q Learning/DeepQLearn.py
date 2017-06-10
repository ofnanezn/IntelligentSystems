import tic_tac_toe as game
import random
import numpy as np
from collections import deque

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD , Adam
import matplotlib.pyplot as plt

N_ACTIONS = 36
GAMMA = 0.9 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.3 # starting value of epsilon
REPLAY_MEMORY = 15000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
LEARNING_RATE = 1e-4

def build_model():
    model = Sequential()
    model.add(Dense(36, input_dim=36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    return model

def train_network(model, args):
    game_state = game.ofttt()
    RM = deque()
    (x_t, r_0, terminal) = (game_state.initial, 0, False)
    s_t = np.array(x_t[1])
    #player2 = game.random_player(game_state, game_state.initial)
    if args == 'run':
        OBSERVE = 999999999    #Keep observe, never train
        epsilon = FINAL_EPSILON    #Use a small epsilon to choose mainly policy actions
        #Load model
        print ("Now we load weight")  
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")
    else:
        #Assign an observation variable and max epsilon to train
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
        
    lossA = []

    t = 0
    
    while(True):
        
        if terminal:
            print "------------------------------------------FINAL-----------------------------------------------"
            print "Reward:",r_t
            game_state.display(x_t)
            (x_t, r_0, terminal) = (game_state.initial, 0, False)
            #game_state = game_state.initial         
        
        #Initialize variables
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([N_ACTIONS])    #Output vector of actions a_[t] = 1 for action to take
        
        player = x_t[0]
        
        if player == 1:
            if random.random() <= epsilon:    #At the first move, choose randomly
                print("----------Random Action----------")
                action_index = random.randrange(N_ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t.reshape(1,36))       #input the state at time t
                moves = game_state.legal_moves(x_t)
                max_Q = np.argmax(q)         #Take the max q value predicted from network
                action_index = 1 + max_Q         #Assign action to the argmax Q
                a_t[max_Q] = 1               #Output vector a_t = 1 for max_Q

            #Decrease epsilon by a smalll factor
            if epsilon > FINAL_EPSILON and t > OBSERVE:               
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 100000
            
            else:
                x_t1, r_t, terminal = game_state.next_state(action_index, x_t) #run the selected action and observed next state and reward
            #print " 1:", r_t, action_index#, x_t[1]
            
            
        else:
            #Random agent
            moves = game_state.legal_moves(x_t)
            randomAction = moves[random.randrange(0, len(moves))]
            x_t1, r_t, terminal = game_state.next_state(randomAction, x_t)
            #print "-1:", r_t, randomAction#, x_t[1]            
            
        s_t1 = np.array(x_t1[1])
        RM.append((s_t, action_index, r_t, s_t1, terminal))    # store the transition in the Replay Memory
        if len(RM) > REPLAY_MEMORY:
            RM.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(RM, BATCH)
            inputs = np.zeros((BATCH, N_ACTIONS))  
            #print (inputs.shape)
            targets = np.zeros((inputs.shape[0], N_ACTIONS)) 

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                x0_t = np.array(minibatch[i][0])
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward
                state_t = x0_t.reshape(1,N_ACTIONS)

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1.reshape(1,N_ACTIONS))
                if action_t == 36:
                    action_t = 35

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            loss += model.train_on_batch(inputs, targets)
            lossA.append(loss)
            
        moves = game_state.legal_moves(x_t)
        #print "Moves: ", moves      
        player = x_t1[0]        
        s_t = s_t1
        x_t = x_t1
        t = t + 1
        
        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            

        # print info
        if t == 10000:
            plt.plot(lossA)
            plt.show()
        info = ""
        if t <= OBSERVE:
            info = "observe"
        else:
            info = "train"

        print("TIMESTEP", t, "/ STATE", info, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
       
    print("Episode finished!")
    print("************************")

def playGame(args):
    model = build_model()
    train_network(model,args)

def main():
    #parser = argparse.ArgumentParser(description='Description of your program')
    #parser.add_argument('-m','--mode', help='Train / Run', required=True)
    #args = vars(parser.parse_args())
    playGame(args="train")

if __name__ == "__main__":
    main()