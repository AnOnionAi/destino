from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import muzero

origins = ["*"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GameData(BaseModel):
    game: str
    player: int
    game_state: list


## Initialize models
mu_tictactoe = muzero.MuZero("tictactoe", use_ray=False)
# Load weights for torch model
checkpoint_path_tictactoe = "masters/tictactoe/model.checkpoint"
mu_tictactoe.load_model(checkpoint_path=checkpoint_path_tictactoe)

mu_chess = muzero.MuZero("chess", use_ray=False)
# Load weights for torch model
checkpoint_path_chess = "masters/chess/model.checkpoint"
mu_chess.load_model(checkpoint_path=checkpoint_path_chess)


async def play_tictactoe(gamedata, onnx_model):
    # TODO - Add Error Handling For Player
    # replay_buffer_path = "models/tictactoe/replay_buffer.pkl"
    # replay_buffer_path = "models/replay_buffer.pkl"
    # If human player is first / mu second, player will be passed in as 2
    # if human player is second, mu first, player will be passed in as 1
    if gamedata.player not in range(1, 3):
        return {"error:": f"The player cannot be {gamedata.player}"}

    player = gamedata.player
    mu_player = player - 1
    board = gamedata.game_state

    game_state = {"to_play": mu_player, "board_int": board}
    next_move = await mu_tictactoe.api_play(
        render=True, opponent="api", game_state=game_state, onnx_model=onnx_model
    )
    rowcol = next_move.split(",")
    row = rowcol[0]
    col = rowcol[1]
    return {"row": row, "col": col}


async def play_chess(gamedata, onnx_model):
    # TODO - Add Error Handling For Player
    player = gamedata.player
    mu_player = player - 1
    game_state = gamedata.game_state
    # Load Model
    next_move = await mu_chess.api_play(
        render=True,
        opponent="human",
        muzero_player=mu_player,
        game_state=game_state,
        onnx_model=onnx_model,
    )

    print(next_move)
    de = next_move[:2]
    to = next_move[-2:]

    return {"de": de, "to": to}


@app.post("/play")
async def game_play(gamedata: GameData):
    if gamedata.game == "tictactoe":
        return await play_tictactoe(gamedata, onnx_model=False)
    elif gamedata.game == "chess":
        return await play_chess(gamedata, onnx_model=False)
    else:
        return {"error:": "The Game you have selected does not exist"}


@app.post("/torch_play")
async def game_torch_play(gamedata: GameData):
    if gamedata.game == "tictactoe":
        return await play_tictactoe(gamedata, onnx_model=False)
    elif gamedata.game == "chess":
        return await play_chess(gamedata, onnx_model=False)
    else:
        return {"error:": "The Game you have selected does not exist"}
