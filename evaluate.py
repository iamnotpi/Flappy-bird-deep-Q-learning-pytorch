import torch
from torch.utils.tensorboard import SummaryWriter
from src.deep_q_network import DeepQNetwork 
from src.flappy_bird import FlappyBird
from src.utils import preprocess
import os

def evaluate_model(num_games=10, checkpoint_path="trained_models/checkpoint_1000000.pt"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    log_dir = os.path.join('tensorboard', 'evaluation')
    writer = SummaryWriter(log_dir)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DeepQNetwork().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    scores = []
    total_steps = 0
    
    for game in range(num_games):
        game_state = FlappyBird()
        image, _, _ = game_state.next_frame(0)
        
        image = preprocess(image[:game_state.screen_width, :int(game_state.base_y)]).to(device)
        state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
        
        terminal = False
        steps = 0
        score = 0
        q_values = []
        
        while not terminal:
            with torch.no_grad():
                prediction = model(state)[0]
                q_values.append(prediction.cpu())
                action = torch.argmax(prediction).item()
            
            next_image, reward, terminal = game_state.next_frame(action)
            
            if reward == 1: score += 1

                
            if not terminal:
                next_image = preprocess(next_image[:game_state.screen_width, :int(game_state.base_y)]).to(device)
                next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
                state = next_state
            
            steps += 1
            total_steps += 1
        
        scores.append(score)
        
        writer.add_scalar('Score/game', score, game)
        writer.add_text('Game Summary', 
                       f'Game {game + 1}: Score = {score}, Steps = {steps}',
                       game)
        
        print(f"Game {game + 1}: Score = {score}")
        
    avg_score = sum(scores) / len(scores)
    print(f"Average Score over {num_games} games: {avg_score:.2f}")

    
    # Log final metrics
    writer.add_hparams(
        {'num_games': num_games},
        {'avg_score': avg_score}
    )
    
    writer.close()
    return avg_score

if __name__ == "__main__":
    evaluate_model(num_games=50)