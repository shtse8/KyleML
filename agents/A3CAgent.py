import numpy as np
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traceback

from memories.SimpleMemory import SimpleMemory
from memories.Transition import Transition
from optimizers.sharedadam import SharedAdam
from utils.PredictionHandler import PredictionHandler
from utils.errors import InvalidAction
from .Agent import Agent, EpisodeReport


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

    def forward(self, state):
        body_output = self.get_body_output(state)
        probs = F.softmax(self.policy(body_output), dim=-1)
        return probs, self.value(body_output)

    def get_body_output(self, state):
        return self.body(state)
    
    def actor(self, state):
        body_output = self.get_body_output(state)
        return F.softmax(self.policy(body_output), dim=-1)

    def critic(self, state):
        return self.value(self.get_body_output(state))


class A3CAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__("a3c", env, **kwargs)

        # Trainning
        self.learningRate: float = kwargs.get('learningRate', .001)

        # Prediction model (the main Model)
        self.network: Network = Network(
            np.product(self.env.observationShape),
            self.env.actionSpace)
        self.network.share_memory()
        self.optimizer = SharedAdam(self.network.parameters(), lr=self.learningRate)
        
        # multiprocessing
        self.queue = mp.Queue()

        # self.network.to(self.device)
        self.addModels(self.network)
  
    def start(self) -> None:
        # parallel training
        # workers = [Worker(i, self) for i in range(1)]
        
        processes = []
        conns = []
        # n_workers = mp.cpu_count() - 1  # one for main thread
        n_workers = mp.cpu_count() // 2
        # n_workers = 1
        for i in range(n_workers):
            parent_conn, child_conn = mp.Pipe(True)
            p = mp.Process(target=self.startWorker, args=(i, child_conn))
            p.start()
            conns.append(parent_conn)
            processes.append(p)
            # time.sleep(1)
        
        self.epochBegin()
        self.beginPhrase()
        # self.network.to(self.device)
        while True:
            # Process messages
            for i, conn in enumerate(conns):
                if conn.poll():
                    message = conn.recv()
                    if message is not None:
                        if isinstance(message, EpisodeReport):
                            self.episodes.value += 1
                            self.history.append(message)
                            self.update()
                            if self.episodes.value % 100 == 0:
                                self.save()
                            if self.episodes.value > self.target_episodes:
                                self.endPhrase()
                                self.epochEnd()
                                self.epochBegin()
                                self.beginPhrase()
                        else:
                            raise Exception("Unknown Message.")
                    else:
                        # termination
                        break
            # print(self.network.state_dict()["body.0.weight"][0])
            time.sleep(0.001)
        for p in processes:
            p.join()

    def startWorker(self, i, conn):
        try:
            worker = A3CWorker(i, self, conn)
            worker.run()
        except Exception as e:
            print("Worker {0} is failed to start: {1}".format(i, e))
            traceback.print_tb(e.__traceback__)


class Worker():
    def __init__(self, manager: Agent, name: str, conn):
        self.name: str = str(name)
        self.manager = manager
        self.conn = conn
        self.env = self.manager.env.getNew()

        self.report = EpisodeReport()

    def beginEpisode(self) -> None:
        self.report = EpisodeReport()
        self.report.start()
        return True  # self.episodes <= self.target_episodes

    def endEpisode(self) -> None:
        # duration = time.perf_counter() - self.episode_start_time
        # loss = self.loss / self.samples
        self.report.end()
        self.conn.send(self.report)

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
                        self.report.invalidMoves += 1
                        # print(actionMask)
                    # finally:
                state = nextState
            self.endEpisode()

    def commit(self, transition: Transition):
        self.report.rewards += transition.reward
        self.report.steps += 1

    def isTraining(self):
        return True


class A3CWorker(Worker):
    def __init__(self, name: str, manager: Agent, conn, **kwargs):
        super().__init__(manager, name, conn)
        
        # Trainning
        self.gamma: float = kwargs.get('gamma', 0.9)
        self.network: Network = Network(
            np.product(self.env.observationShape),
            self.env.actionSpace)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        # self.optimizer = SharedAdam(self.network.parameters(), lr=0.001)
        self.network.load_state_dict(self.manager.network.state_dict())

        # Memory
        self.memory_size: int = kwargs.get('memory_size', 10000)
        self.memory: SimpleMemory = SimpleMemory(self.memory_size)
        self.n_steps: int = 10
        self.target_episodes: int = 10000

    def beginEpisode(self) -> None:
        self.memory.clear()
        return super().beginEpisode()

    def getPrediction(self, state):
        self.network.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).view(1, -1)
            prediction = self.network.actor(state).squeeze(0)
            return prediction.cpu().detach().numpy()

    def getAction(self, prediction, mask = None):
        handler = PredictionHandler(prediction, mask)
        return handler.getRandomAction() if self.isTraining() else handler.getBestAction()

    def commit(self, transition: Transition) -> None:
        super().commit(transition)
        if self.isTraining():
            self.memory.add(transition)
            if transition.done or self.report.steps % self.n_steps == 0:
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

        batch = self.memory
        if len(batch) == 0:
            return

        states = np.array([x.state for x in batch])
        states = torch.FloatTensor(states).to(self.device).view(states.shape[0], -1)
        
        actions = np.array([x.action for x in batch])
        actions = torch.LongTensor(actions).to(self.device)
        
        action_probs, values = self.network(states)
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
        entropy_loss = -dist.entropy().mean()
        actor_loss = -(dist.log_prob(actions) * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        # value_loss = nn.MSELoss()(values, discountRewards)
        
        total_loss = actor_loss + 0.01 * entropy_loss + value_loss
        
        self.manager.optimizer.zero_grad()
        total_loss.backward()
        self.network.cpu()
        for network, managerNetwork in zip(self.network.parameters(), self.manager.network.parameters()):
            # print(network.grad.data)
            managerNetwork._grad = network.grad
        self.manager.optimizer.step()
        self.network.to(self.device)
        self.network.load_state_dict(self.manager.network.state_dict())

        # Stats
        self.report.trained(total_loss.item(), len(batch))

        # clear memory
        self.memory.clear()
