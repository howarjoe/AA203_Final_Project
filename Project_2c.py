#Code is not super clean and contains several artifacts from earlier versions of the project
#Not meant to be shared as an example, rather as documentation for whwat was done
#I didn't know we'd be asked to submit the code until I was submitting my assignment on the final day, so I didn't get a chance to clean it up
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpy.random as rand
import matplotlib.pyplot as plt

import time as tm
startTime = tm.time()

#Needed Data
Start_Cadence = 175
No_Layers = 6
No_Products = 3
ST = 3
tmax = 10000
t_episode = 1000
Train_Frequency = 100

LayerData = pd.read_csv('LayerData.csv')

LotStarter = pd.read_csv('StarterLotStatus.csv')

W1_Yield = pd.read_csv('Workstation1Yield.csv')

W1_LayerData = LayerData[LayerData["Photo Workstation"] == 1] 

W1_Layers = W1_LayerData["Layer"]

W1_Tools = 2

def NNConverter(states, tool):
    #We're going to fit just based on the tool
    counts, means, Yield, Machine = states
    #Everything needs to be 1D
    counts_flat = counts.flatten()
    counts_flat = counts_flat / np.max((1, np.sum(counts_flat)))
    means_flat = means.flatten()
    means_flat = means_flat / np.max((1, np.sum(means_flat)))
    
    #Machine Setup we'll need to extract setup as a one-hot vector
    layer, product = Machine['Setup Type'][tool]
    Matrix_One_Hot = np.zeros((No_Layers, No_Products))
    Matrix_One_Hot[layer, product] = 1
    Setup_One_Hot = Matrix_One_Hot.flatten()
    
    #Get Yield of each level
    Tool_Yield = Machine['Yield'][tool]
    
    #Concatenate all of these vectors
    con = np.concatenate((counts_flat, means_flat, Yield, Setup_One_Hot, Tool_Yield))
    return con.reshape(1, 66)

#%% Functions to complete operation
# This is how we start new material
def startAction(n=5):
    prodIDs = []
    for i in range(n):
        prodIDs.append(rand.choice([0, 1, 2], p=[0.6, 0.2, 0.2]))
    return prodIDs
#Relic from when there were multiple WS before trimming scope
levelWS = LayerData['Photo Workstation'].to_list()
levelRunning = LayerData['Remaining Time'].to_list()
levelPhoto = LayerData['Photo Time'].to_list()
#Will be used in implementation of state
def removeMax(arr):
    new_arr = arr.copy()
    new_arr.remove(max(arr))
    if new_arr == arr:
        raise Exception("Yeah, you didn't change anything")
    return new_arr
def removeZeros(arr):
    new_arr = arr.copy()
    new_arr = [i for i in new_arr if i!=0]
    return new_arr

#WIP is an array of LayerXProduct, it is a list of time waiting for every lot at step
#Running is a list of remaining times at non-photo steps
#Machines shows everything about machines, current tool yield, layer currently set up for, time remaining (idle if 0)

#Generate empty WIP/Running
WIP = []
Running = []
for i in range(No_Layers):
    WIP_Row = []
    Running_Row = []
    for j in range(No_Products):
        WIP_Row.append([])
        Running_Row.append([])
    WIP.append(WIP_Row)
    Running.append(Running_Row)
WIP_Init = list(WIP)
Running_Init = list(Running)


#Count zeros works for both WIP and running
def countZeros(lists):
    arr = np.zeros((No_Layers, No_Products))
    for i in range(No_Layers):
        for j in range(No_Products):
            arr[i, j] = lists[i][j].count(0)
            lists[i][j] = removeZeros(lists[i][j])
    return arr



#Increment List of Lists
#Or decrement used in simulation
def inc_dec_choice(lists, k = 1):
    for i in range(No_Layers):
        for j in range(No_Products):
            single = lists[i][j]
            single = [x+k for x in single]
            lists[i][j] = single
    return lists

#Generate Initial Tool States
Yield_Tools = [[1, 0.95, 1, 1, 0.95, 1],
         [1, 1, 1, 1, 1, 0.95]]
W1_State = {}
time = []
setup = []
for tool in range(W1_Tools):
    time.append(0)
    setup.append([0, 0])
W1_State['Time Remaining'] = time
W1_State['Setup Type'] = setup
W1_State['Yield'] = Yield_Tools

W1_State_Init = dict(W1_State)

#Generate Initial Yield State
S_Yield = []
for i in range(No_Layers):
    S_Yield.append(1)
S_Yield_Init = list(S_Yield)

#Helper functions for input to NN
def Average(lst):
    if len(lst) == 0:
        return 0
    else:
        return sum(lst) / len(lst)

def WIPCounts_Averages(lists):
    counts = np.zeros((No_Layers, No_Products))
    averages = np.zeros((No_Layers, No_Products))
    for i in range(No_Layers):
        for j in range(No_Products):
            single = lists[i][j]
            counts[i, j] = len(single)
            averages[i, j] = Average(single)
    return counts, averages
   

#Rewards
def Reward(states):
    counts, means, Yield, Machine = states
    multiplied = np.multiply(counts, means)
    return -1 * np.sum(multiplied)
def WIPSigma(states):
    counts, means, Yield, Machine = states
    multiplied = np.multiply(counts, means)
    return -1 * np.std(multiplied)

def TrueReward(states, w1=1, w2 = 1500, w3 = 1):
    counts, means, Yield, Machine = states
    WIP_Comp = Reward(states)
    Yield_Comp = np.prod(Yield)
    return w1*WIP_Comp + w2*Yield_Comp

#The order of operations in the timestep is incorrect
#We should do the action first and then increment/decrement etc.
def timeStep_2(t, WIP, Running, Machine, Yield, actions):
    #We start with the action phase
    
    counts, means = WIPCounts_Averages(WIP)
    
    #Find all available tools and generate an action for each
    States = (counts, means, Yield, Machine)
    idlePenalty = 0
    switchingPenalty = 0
    
    for i in range(len(Machine['Time Remaining'])):
        if Machine['Time Remaining'][i] == 0:
            
            action = actions[i]
            if action:
                #Update Yield of Layer in Yield State
                layer, product = action
                
                Yield[layer] = Yield[layer] * 0.95 + 0.05 * Machine['Yield'][i][layer]
                
                #Perform dispatching action
                if action == Machine['Setup Type'][i]:
                    switch = 0
                else:
                    switch = ST
                    switchingPenalty -= 6000
                    Machine['Setup Type'][i] = action
                processTime = levelPhoto[layer] + switch
                Machine['Time Remaining'][i] = processTime
                WIP[layer][product] = removeMax(WIP[layer][product])
            else:
                idlePenalty -= 10000

    #Now we're going to get ourselves ready for the next time step
    #First we're going to increment all waiting times
    WIP = inc_dec_choice(WIP)
    #And decrement all running times
    Running = inc_dec_choice(Running, k=-1)
    
    #Next we're going to look for everything that's ready to track out of Running
    #As well as removing it in one line... Be careful with this function
    forWIP = countZeros(Running)

    #Now we want to get all of these from Running to WIP
    #If it's the final layer, it just goes away...
    for i in range(No_Layers-1):
        for j in range(No_Products):
            for l in range(int(forWIP[i, j])):
                WIP[i+1][j].append(0)
                
    #Then we're going to start new lots on a cadence:
    if t%Start_Cadence == 0:
        startedLots = startAction(n=5)
        for j in range(No_Products):
            count = startedLots.count(j)
            for l in range(count):
                WIP[0][j].append(0)
    #Need to insert sampling Decision
    #Add time to process Time
        
    #Then we're going to look for everything that's ready to track out of photo
    #We'll append them to Running
    #And we'll decriment everything while we're at it

    for i in range(len(Machine['Time Remaining'])):
        if Machine['Time Remaining'][i] == 1:
            layer, product = Machine['Setup Type'][i]
            processTime = levelRunning[layer]
            Running[layer][product].append(processTime)
        if Machine['Time Remaining'][i] > 0:
            Machine['Time Remaining'][i] += -1
    
    Counts_Prime, Avg_Prime = WIPCounts_Averages(WIP)
    States_Prime = (Counts_Prime, Avg_Prime, Yield, Machine)
    
    #Calculate Reward for new state
    reward = TrueReward(States_Prime)
    
    return WIP, Running, Machine, Yield, States_Prime, reward


#Heuristic Policy
def Heuristic(states):
    counts, means, Yield, Machine = states
    
    actions = []
    for m in range(W1_Tools):
        if Machine['Time Remaining'][m] != 0:
            actions.append([])
        else:
            layer, product = Machine['Setup Type'][m]
            if counts[layer, product] > 0:
                action = Machine['Setup Type'][m]
            else:
                action = list(np.unravel_index(counts.argmax(), counts.shape))
                count_in_max = counts.max()
                if count_in_max == 0:
                    action = []
            if action:
                layer, product = action.copy()
                #Can't stage one lot to more than one tool
                counts[layer, product] += -1
            actions.append(action)
    return actions


#Test simulation with Heuristic
#What are the KPIs that we can plot?
#Count at each level
#Wait Time by product/level
Reward_Heur = np.zeros(tmax)
True_Reward_Heur = np.zeros(tmax)

WIP_By_Level = np.zeros((tmax, No_Layers))
Running_By_Level = np.zeros((tmax, No_Layers))
Yield_By_Time = np.zeros(tmax)
counts, means = WIPCounts_Averages(WIP)

#Find all available tools and generate an action for each
States = (counts, means, S_Yield, W1_State)

Reward_Benchmark = 0

for t in range(tmax):
    
    WIP, Running, W1_State, S_Yield, States_Prime, reward = timeStep_2(t, WIP, Running, W1_State, S_Yield, Heuristic(States))
    States = States_Prime
    if t==1:
        input_size = len(NNConverter(States_Prime, 0)[0])
    
    Yield_Prod = np.prod(S_Yield)
    Yield_By_Time[t] = Yield_Prod
    Counts = States_Prime[0]
    Layer_Sum = np.sum(Counts, axis=1)
    RunningCount, RunningAvg = WIPCounts_Averages(Running)
    Running_Sum = np.sum(RunningCount, axis=1)
    WIP_By_Level[t, :] = Layer_Sum
    Running_By_Level[t, :] = Running_Sum
    Reward_Heur[t] = Reward(States_Prime)
    True_Reward_Heur[t] = reward
    
    if t>1000 and t<2001:
        Reward_Benchmark += reward

    #print(t, WIP)
    #print(t, Running)
#plt.plot(range(tmax), Reward_Heur)
for i in range(No_Layers):
    plt.plot(range(tmax), WIP_By_Level[:, i], label='Layer='+str(i))
plt.title('WIP Level Over Time')
plt.legend()
plt.show()
plt.close()

plt.plot(range(tmax), Reward_Heur)
plt.title('Total Fab WIP Over Time')
plt.show()
plt.close()

plt.plot(range(tmax), True_Reward_Heur)
plt.title('Reward Over Time')
plt.show()
plt.close()

plt.plot(range(tmax), Yield_By_Time)
plt.title('Yield')
plt.show()
plt.close()

print('Reward Sum for Heuristic was: ', np.sum(True_Reward_Heur))


#%% Now that we've shown performance on Heuristic we move on to DQN
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#Building Neural Net to Approximate q for each state-action combination
#First we're going to need a function that takes our state and turns it into something that a Neural Net can work with
#I actually moved this above

#Using Keras. Move to Jax if time allows
    
#Action space consists of all possible product/layer combinations +1 for no action
actionSize = No_Layers * No_Products + 1
#input_dim set above through Heuristic loop
DQN = Sequential()
DQN.add(Dense(72, input_dim = input_size, activation = 'relu'))
DQN.add(Dense(32, activation='relu'))
DQN.add(Dense(actionSize, activation = 'linear'))
DQN.compile(optimizer=Adam(), loss = 'mse')

N = 40
gamma = 0.99
epsilon = 1
minibatchSize = 30
cumReward = []  #Store this to plot
replayMemory = [] #Hold state transitions
memorySize = 100000

#We're going to need a function that can go from an integer to layerxprod
def IntToIndex(integer):
    if integer == actionSize-1:
        return []
    else:
        return list(np.unravel_index(integer, (No_Layers, No_Products)))

#And one that goes the other way
def IndexToInt(Index):
    if Index == []:
        return actionSize-1
    else:
        i, j = Index
        return i*No_Products + j
#We're going to need a function to find all available actions...
def AllAvailableActions(states, tool):
    #Use same format as Neural Network
    #First Check if tool is idle
    actionList = [[]]
    counts, means, Yield, Machine = states
    if Machine['Time Remaining'][tool] > 0:
        return [[]]
    else:
        if np.max(counts) == 0:
            return [[]]
        else:
            for i in range(No_Layers):
                for j in range(No_Products):
                    if counts[i, j] > 0:
                        actionList.append([i, j])
            return actionList

#%% Need a function to  train our neural network to approximate q correctly
def train(replayMemory, minibatchSize=32):
    # choose <s,a,r,s'> experiences randomly from the memory
    minibatch = np.random.choice(replayMemory, minibatchSize, replace=True)
    #Need lists for state transitions so we can get minitbatch created
    sArray =      np.array(list(map(lambda x: x['s'], minibatch)))
    aArray =      np.array(list(map(lambda x: x['a'], minibatch)))
    rArray =      np.array(list(map(lambda x: x['r'], minibatch)))
    sprimeArray = np.array(list(map(lambda x: x['sprime'], minibatch)))
    qvals_sprime = []
    fittedVals = []
    for i in range(minibatchSize):
        #Need q values of every prime state
        qvals_sprime.append(model.predict(sprimeArray[i].reshape(1, 66), verbose=0))
        fittedVals.append(model.predict(sArray[i].reshape(1, 66), verbose=0))
    
    for i,(s,a,r,qvals_sprime) in enumerate(zip(sArray,aArray,rArray,qvals_sprime)): 
        #Removed original condition to check if terminal
        #We want to fit the qvals to this
        #And obviously we are only altering the NN output of the actions we actually took
        target = r + gamma * np.max(qvals_sprime)
        fittedVals[i][0][a] = target
    #"X" Values for the neural network
    X_NN = []
    for i in range(minibatchSize):
        X_NN.append(sArray[i].reshape(1, 66))
    
    model.fit(X_NN, fittedVals, epochs=1, verbose=0)
    return model

#Honestly don't start empty for WIP either

Train_Counter = 0
#Now we need to interact with the environment again
for n in range(N):
    #Reset Everything
    #Reset Everything
    #Simulate 1000 steps to put the WIP in a good spot
    WIP = []
    Running = []
    for i in range(No_Layers):
        WIP_Row = []
        Running_Row = []
        for j in range(No_Products):
            WIP_Row.append([])
            Running_Row.append([])
        WIP.append(WIP_Row)
        Running.append(Running_Row)
    S_Yield = []
    for i in range(No_Layers):
        S_Yield.append(1)
    W1_State = {}
    time = []
    setup = []
    for tool in range(W1_Tools):
        #Start Tools in random setup state
        time.append(0)
        if n== 0:
            setup.append([0, 0])
        else:
            random_int = np.random.randint(0, actionSize-1)
            random_action = IntToIndex(random_int)
            setup.append(random_action)
            
    W1_State['Time Remaining'] = time
    W1_State['Setup Type'] = setup
    W1_State['Yield'] = Yield_Tools
    
    States = (counts, means, S_Yield, W1_State)
    counts, means = WIPCounts_Averages(WIP)
    for t in range(1000):
         
        WIP, Running, W1_State, S_Yield, States_Prime, reward = timeStep_2(t, WIP, Running, W1_State, S_Yield, Heuristic(States))
        States = States_Prime   

    t = 0
    sumReward = 0
    
    while t < t_episode:
        #Feedforward pass to get predicted q-values for all actions
        #We're training for each tool individually
        #Action List to Feed to simulations
        Actions = []
        #Actions that I actually want to train on (if there are no options we don't want to train on that)
        Train_Actions = []
        #Corresponding States I actually want to train on
        Train_States = []
        Train_Tools = []
        
        
        for m in range(W1_Tools):
            
            actionList = AllAvailableActions(States, m)
            S_For_NN = NNConverter(States, m)
            if np.random.random() < epsilon:
                choice = np.random.randint(0, len(actionList))
                action = actionList[choice]
            else:
                qvals_s = DQN.predict(S_For_NN, verbose=0)
                qvals_s = qvals_s[0]
                qVals_a = []
                for a in range(len(actionList)):
                    integer = IndexToInt(actionList[a])
                    qVals_a.append(qvals_s[integer])
                maxQ = np.argmax(qVals_a)
                action = actionList[maxQ]
            
            Actions.append(action)
            if actionList != [[]]:
                #Only want to train if the tools were given a choice. Otherwise we bias the non-action action
                integer = IndexToInt(action)
                Train_Actions.append(integer)
                Train_States.append(S_For_NN[0])
                Train_Tools.append(m)
            #Update SS for any actions taken
            if action:
                layer, product = action
                States[0][layer, product] -= 1
        #Take TimeStep
        WIP, Running, W1_State, S_Yield, States_Prime, reward = timeStep_2(t, WIP, Running, W1_State, S_Yield, Actions)
        
        sumReward += reward
        if len(replayMemory) > memorySize:
            replayMemory.pop(0)
        
        for a in range(len(Train_Actions)):
            Train_SPrime = NNConverter(States_Prime, Train_Tools[a])[0]
            replayMemory.append({"s": Train_States[a], "a": Train_Actions[a], "r": reward, "sprime": Train_SPrime})
            Train_Counter += 1
            if Train_Counter % Train_Frequency == 0:
                model = train(replayMemory, minibatchSize = minibatchSize)
                if epsilon > 0.01:
                    epsilon -= 0.005
                    print(epsilon)
        
        States = States_Prime
        t += 1
        if t%100 == 0:
            print("Episode: ", n, " Time: ", t, "Elapsed Real Time: ", tm.time() - startTime)
        
    print("Total Reward for ", n, ": ", sumReward)
    cumReward.append(sumReward)
        
print(cumReward)
print(np.sum(True_Reward_Heur))    

#%%
#Now simulate with Neural Net
def TrainedDQN(states):
    Actions = []
    for m in range(W1_Tools):
        S_For_NN = NNConverter(States, m)

        actionList = AllAvailableActions(States, m)
        if actionList == [[]]:
            action = []
        else:
            qVals_a = []
            qvals_s = model.predict(S_For_NN, verbose=0)
            for a in range(len(actionList)):
                integer = IndexToInt(actionList[a])
                qVals_a.append(qvals_s[0][integer])
    
            maxQ = np.argmax(qVals_a)
            action = actionList[maxQ]
            if action:
                layer, product = action
                States[0][layer, product] -= 1
        Actions.append(action)
    return Actions


#Do equivalent as we did for Heuristic
Reward_DQN = np.zeros(tmax)
True_Reward_DQN = np.zeros(tmax)

WIP_By_Level = np.zeros((tmax, No_Layers))
Running_By_Level = np.zeros((tmax, No_Layers))
Yield_By_Time = np.zeros(tmax)



#Reset Everything
WIP = []
Running = []
for i in range(No_Layers):
    WIP_Row = []
    Running_Row = []
    for j in range(No_Products):
        WIP_Row.append([])
        Running_Row.append([])
    WIP.append(WIP_Row)
    Running.append(Running_Row)
S_Yield = []
for i in range(No_Layers):
    S_Yield.append(1)
W1_State = {}
time = []
setup = []
for tool in range(W1_Tools):
    time.append(0)
    setup.append([0, 0])
W1_State['Time Remaining'] = time
W1_State['Setup Type'] = setup
W1_State['Yield'] = Yield_Tools

counts, means = WIPCounts_Averages(WIP)
States = (counts, means, S_Yield, W1_State)

#Honesty need to turn the tmax simulation into a function to call... if I get a chance

for t in range(tmax):
    Action = TrainedDQN(States)
    WIP, Running, W1_State, S_Yield, States_Prime, reward = timeStep_2(t, WIP, Running, W1_State, S_Yield, TrainedDQN(States))

    States = States_Prime
    if t%100 == 0:
        print("Current Time in Simulation ", t)
    
    
    Yield_Prod = np.prod(S_Yield)
    Yield_By_Time[t] = Yield_Prod
    Counts = States_Prime[0]
    Layer_Sum = np.sum(Counts, axis=1)
    RunningCount, RunningAvg = WIPCounts_Averages(Running)
    Running_Sum = np.sum(RunningCount, axis=1)
    WIP_By_Level[t, :] = Layer_Sum
    Running_By_Level[t, :] = Running_Sum
    Reward_DQN[t] = Reward(States_Prime)
    True_Reward_DQN[t] = reward



for i in range(No_Layers):
    plt.plot(range(tmax), WIP_By_Level[:, i], label='Layer='+str(i))
plt.title('WIP Level Over Time - DQN')
plt.legend()
plt.show()
plt.close()

plt.plot(range(tmax), Reward_DQN)
plt.title('Total Fab WIP Over Time - DQN')
plt.show()
plt.close()

plt.plot(range(tmax), True_Reward_DQN)
plt.title('Reward Over Time - DQN')
plt.show()
plt.close()

plt.plot(range(tmax), Yield_By_Time)
plt.title('Yield - DQN')
plt.show()
plt.close()

print('Reward Sum for DQN was: ', np.sum(True_Reward_DQN))
