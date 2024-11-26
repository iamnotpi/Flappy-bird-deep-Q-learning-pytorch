"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import preprocess

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def test(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load("{}/checkpoint_1000000.pt".format(opt.saved_path), map_location=device)
    model = DeepQNetwork()
    model.load_state_dict(ckpt['model_state_dict'])
    game_state = FlappyBird()
    image, _, _ = game_state.next_frame(0)
    image = preprocess(image[:game_state.screen_width, :int(game_state.base_y)])
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()

        next_image, _, _ = game_state.next_frame(action)
        next_image = preprocess(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        state = next_state


if __name__ == "__main__":
    opt = get_args()
    test(opt)