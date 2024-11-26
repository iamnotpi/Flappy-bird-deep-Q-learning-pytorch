from dataclasses import dataclass
import os
import random
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import transforms

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird


@dataclass
class Args: 
    history_length: int = 4
    gamma: float = 0.99
    batch_size: int = 32
    target_iter: int = 2500 # Update every 10000 frames
    explore_iter: int = 10000 # Update weights after 40000 frames
    max_iter: int = 1000000 # a.k.a. 4000000 frames
    eps_init: float = 0.1
    eps_final: float = 1e-4
    replay_memory_size: int = 50000
    lr: float = 2.5e-4


class ReplayBuffer: 
    def __init__(self, capacity, history_length, device): 
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros((capacity, history_length, 84, 84))
        self.next_states = torch.zeros((capacity, history_length, 84, 84))
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self.terminals = torch.zeros(capacity, dtype=torch.bool)
        self.pos = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, terminal):
        self.states[self.pos] = state.cpu()
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state.cpu()
        self.terminals[self.pos] = terminal
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,))
        return (
            self.states[indices].to(self.device, non_blocking=True),
            self.actions[indices].to(self.device, non_blocking=True),
            self.rewards[indices].to(self.device, non_blocking=True),
            self.next_states[indices].to(self.device, non_blocking=True),
            self.terminals[indices].to(self.device, non_blocking=True)
        )
        

os.makedirs('trained_models', exist_ok=True)
args = Args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

policy_model = DeepQNetwork().to(device)
target_model = DeepQNetwork().to(device)
print(f'Number of parameters: {sum(p.numel() for p in policy_model.parameters())}')

optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr, fused=(device=='cuda'))
replay_buffer = ReplayBuffer(args.replay_memory_size, args.history_length, device)

iter = 0
game_state = FlappyBird()
frame_counter = 0
skip_frames = args.history_length

def preprocess(image):
    img = rgb_to_grayscale(transforms.Resize((84, 84))(
        torch.from_numpy(image[:game_state.screen_width, :int(game_state.base_y)]).permute(2, 1, 0)
    ))
    threshold = 1 
    binary_img = (img < threshold).float()
    return binary_img

image, reward, terminal = game_state.next_frame(0)
image = preprocess(image)
state = torch.cat([image for _ in range(args.history_length)], dim=0)[None, ...]    

if device == 'cuda': 
    torch.backends.cudnn.benchmark = True

while iter < args.max_iter:
    policy_model.eval()
    # Update target model's weights 
    if iter % args.target_iter == 0: 
        target_model.load_state_dict(policy_model.state_dict())

    # Update replay memory (online learning paradigm)
    # Take action based on policy net
    # Get Q value for each action
    explore = iter < args.explore_iter

    state = state.to(device)
    if explore:
        frame_counter = (frame_counter + 1) % skip_frames
        action = random.randint(0, 1) if frame_counter == 0 else 0
    else: 
        with torch.no_grad():
            q_values = policy_model(state).squeeze()
        # Select action using epsilon-greedy
        eps = args.eps_init + (args.eps_final - args.eps_init) * iter / args.max_iter
        action = random.randint(0, 1) if random.random() < eps else torch.argmax(q_values).item() 

    next_image, reward, terminal = game_state.next_frame(action)

    # Keep the last 4 frames of the history 
    next_state = torch.cat([state[0, 1:], preprocess(next_image[:game_state.screen_width, :int(game_state.base_y)]).to(device)])[None, ...]
    replay_buffer.push(state, action, reward, next_state, terminal)

    if replay_buffer.size >= args.batch_size and not explore:
        # Train policy net
        policy_model.train()
        states, actions, rewards, next_states, terminals = replay_buffer.sample(args.batch_size)

        cur_q_values = policy_model(states) # (B, 2)
        # Only update q values for taken actions
        q_values = cur_q_values[range(len(actions)), actions]
        with torch.no_grad(): 
            next_q_values = target_model(next_states) # (B, 2)
        
        target_q_values = rewards + args.gamma * (~terminals) * next_q_values.max(dim=1)[0]
        optimizer.zero_grad()
        loss = F.mse_loss(q_values, target_q_values)
        loss.backward()
        optimizer.step()
        state = next_state
        if (iter + 1) % 10000 == 0: 
            checkpoint = {
                'model_state_dict': policy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
                'epsilon': eps
            }
            torch.save(checkpoint, f'trained_models/checkpoint_{iter + 1}.pt')
    iter += 1