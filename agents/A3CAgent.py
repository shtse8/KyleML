import numpy as np
import time
from memories.SimpleMemory import SimpleMemory
from memories.Transition import Transition
from .Agent import Agent
from utils.PredictionHandler import PredictionHandler
from utils.errors import InvalidAction
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp


class Network(nn.Module):
    def __init__(self, n_inputs, n_outputs, name="default"):
        super().__init__()
        self.name = name

        hidden_nodes = 128
        self.body = nn.Sequential(
            nn.Linear(n_inputs, hidden_nodes),
            nn.ReLU())
            
        # Define policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, n_outputs))
            
        # Define value head
        self.value = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1))

    def predict(self, state):
        body_output = self.get_body_output(state)
        probs = F.softmax(self.policy(body_output), dim=-1)
        return probs, self.value(body_output)

    def get_body_output(self, state):
        return self.body(state)
    
    def critic(self, state):
        return self.value(self.get_body_output(state))


class A3CAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__("a3c", env, **kwargs)

        # Trainning
        self.learningRate: float = kwargs.get('learningRate', .001)

        # Prediction model (the main Model)
        self.network: Network = Network(
            np.product(self.env.observationSpace),
            self.env.actionSpace)
        self.network.share_memory()
        self.optimizer = SharedAdam(self.network.parameters(), lr=self.learningRate)
        
        # multiprocessing
        self.queue = mp.Queue()
        self.episodes = mp.Value('i', 0)

        # self.network.to(self.device)
        self.addModels(self.network)
  
    def start(self) -> None:
        # parallel training
        # workers = [Worker(i, self) for i in range(1)]
        while self.beginPhrase():
            processes = []
            for i in range(mp.cpu_count() //     2):
            # for i in range(4):
                p = mp.Process(target=self.startWorker, args=(i,))
                p.start()
                processes.append(p)
                # time.sleep(1)
            while True:
                r = self.queue.get()
                if r is not None:
                    self.rewardHistory.append(r)
                    self.update()
                    # print(self.episodes.value, r)
                else:
                    # termination
                    break
            for p in processes:
                p.join()
            self.endPhrase()

    def startWorker(self, i):
        try:
            worker = Worker(i, self)
            worker.run()
        except Exception as e:
            print("Worker {0} is failed to start: {1}".format(i, e))

    def report(self, loss: float, samples: int):
        self.loss += loss * samples
        self.samples += samples


class Worker():
    def __init__(self, name: str, manager, **kwargs):
        super(Worker, self).__init__()
        self.name: str = str(name)
        self.manager = manager

        # Trainning
        self.gamma: float = kwargs.get('gamma', 0.9)
        self.env = self.manager.env.getNew()
        self.network: Network = Network(
            np.product(self.env.observationSpace),
            self.env.actionSpace)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        # Memory
        self.memory_size: int = kwargs.get('memory_size', 10000)
        self.memory: SimpleMemory = SimpleMemory(self.memory_size)
        self.n_steps: int = 50
        self.target_episodes: int = 10000

        self.episodes = 0
        self.rewards = 0
        self.steps = 0
        self.invalidMoves: int = 0

        print("Worker", self.name)

    def isTraining(self):
        return True

    def run(self):
        while self.beginEpisode():
            state = self.env.reset()
            done: bool = False
            while not done:
                reward: float = 0.
                nextState = state
                actionMask = np.ones(self.env.actionSpace)
                prediction = self.getPrediction(state)
                while True:
                    try:
                        action = self.getAction(prediction, actionMask)
                        nextState, reward, done = self.env.takeAction(action)
                        self.commit(Transition(state, action, reward, nextState, done))
                        break
                    except InvalidAction:
                        actionMask[action] = 0
                        self.invalidMoves += 1
                        # print(actionMask)
                    # finally:
                state = nextState
            self.endEpisode()

    def getPrediction(self, state):
        self.network.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).view(1, -1)
            prediction = self.network.predict(state)[0].squeeze(0)
            return prediction.cpu().detach().numpy()

    def getAction(self, prediction, mask = None):
        handler = PredictionHandler(prediction, mask)
        return handler.getRandomAction() if self.isTraining() else handler.getBestAction()

    def beginEpisode(self) -> None:
        self.memory.clear()
        self.episodes += 1
        self.episode_start_time = time.perf_counter()
        self.rewards = 0
        self.steps = 0
        self.loss = 0
        self.samples = 0
        return self.episodes <= self.target_episodes

    def endEpisode(self) -> None:
        
        with self.manager.episodes.get_lock():
            self.manager.episodes.value += 1
        self.manager.queue.put(self.rewards)
        # print(self.name, self.rewards)
        # self.rewardHistory.append(self.rewards)
        # self.stepHistory.append(self.steps)
        # if self.isTraining():
        #     self.lossHistory.append(self.loss / self.samples)
        # self.update()
        pass

    def commit(self, transition: Transition):
        self.rewards += transition.reward
        self.steps += 1
        if self.isTraining():
            self.memory.add(transition)
            if transition.done or self.steps % self.n_steps == 0:
                self.learn()

    def getDiscountedRewards(self, rewards, gamma, finalReward):
        discountRewards = np.zeros_like(rewards).astype(float)
        runningReward = finalReward
        for i in reversed(range(len(rewards))):
            runningReward = runningReward * gamma + rewards[i]
            discountRewards[i] = runningReward
        return discountRewards

    def learn(self) -> None:
        self.network.train()

        batch = self.memory.getLast(self.n_steps)
        if len(batch) == 0:
            return

        states = np.array([x.state for x in batch])
        states = torch.FloatTensor(states).to(self.device).view(states.shape[0], -1)
        
        actions = np.array([x.action for x in batch])
        actions = torch.LongTensor(actions).to(self.device)
        
        action_probs, values = self.network.predict(states)
        values = values.squeeze(1)

        # with torch.no_grad():
        rewards = np.array([x.reward for x in batch])
        finalReward = 0
        if not batch[-1].done:
            nextState = torch.FloatTensor(batch[-1].nextState).to(self.device).view(1, -1)
            finalReward = self.network.critic(nextState).item()
        targetValues = self.getDiscountedRewards(rewards, self.gamma, finalReward)
        targetValues = torch.FloatTensor(targetValues).to(self.device)
        advantages = targetValues - values
        
        dist = torch.distributions.Categorical(probs=action_probs)
        entropy_loss = dist.entropy()
        actor_loss = -(dist.log_prob(actions) * advantages.detach() + entropy_loss * 0.005)
        value_loss = advantages.pow(2).mean()
        # value_loss = nn.MSELoss()(values, discountRewards)
        
        total_loss = (actor_loss + value_loss).mean()
        
        self.manager.optimizer.zero_grad()
        total_loss.backward()
        self.network.cpu()
        for network, managerNetwork in zip(self.network.parameters(), self.manager.network.parameters()):
            managerNetwork._grad = network.grad
        # nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), 0.5)
        self.manager.optimizer.step()
        self.network.to(self.device)
        
        # pull global parameters
        self.network.load_state_dict(self.manager.network.state_dict())

        # Stats
        self.manager.report(total_loss.item(), len(batch))


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr=lr)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
