#     <h1 align="center">                      The Destiny Of Mu0 </h1>

![alt text](https://assets.pokemon.com/assets/cms2/img/pokedex/full/151.png)

To get started with Muz. Use Poetry. 
```
poetry install 
```

If you do not have Poetry, run this command. (Linux Only. Destino does not work on Windows) 

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```
#### To Run mu

```
poetry run python main.py
```

#### Save models in the Cloud (Will Have to Setup. Uses Google Cloud)
Change cloud_save equal to True in games/config/game_name.yaml.

#### To Run the Ai as an API. 

```
poetry run uvicorn app:app --reload
```

#### To See Graphs On Tensorboard
```
poetry run tensorboard --logdir ./results
```
### Google Colab
##### Configure only use 2 workers in game yaml file. 1 Self Player and 1 Reanalyze Worker  1 Worker per 1 Thread. 
Copy the vars.env file into Google Cloud Drive. Enter the credentials for your Github Account.
Open destino.ipynp on Google Co-lab to use with GPU for free. 
Two additional notebooks are available for educational purposes. 

#### Python version need to match up between all programs for ray, poetry & colab to work.

### How to Create a Game Wheel - Python Package
### [Zarena Repo](https://github.com/ZetiAi/zarena) For Rust Versions of Gato, Chess, Checkers, Blackjack and Poker. 
# Checkout notebooks/build_game_wheel.ipynp
```
!poetry install
!apt install rustc
!poetry run maturin build --release
```
# The Build System
## Rust Only 

Cargo.toml -> Rust Build

Uses Cargo 

cargo run 
cargo check 
cargo build 
cargo build --release 

## Python & Rust Hybrid - Building the Games For Destino
#### Maturin 

Will need to change pyproject.toml build system to 
 "maturin>=0.10,<0.11"


When a game package is made with maturin. We want the tar.gz file in games/gym
Insert into the pyproject.toml file like this:

gym_chess = { path = "games/gyms/gym_chess-0.3.0.tar.gz" }

To specify python dependencies, add a list requires-dist in a [package.metadata.maturin] section in the Cargo.toml. This list is equivalent to install_requires in setuptools:

#### Rust&Python - Dev 
```
maturin build
```
#### Rust&Python - Production
```
maturin build --release 
```
Maturin builds the wheels and stores them in a folder (target/wheels by default)
Maturin develop builds the crate and installs it as a python module directly in the current virtualenv. 

#### Diagnose Models
The following Debian packages must be installed in order to use this diagnose models tool. 
```
sudo apt-get install graphviz
sudo apt-get install xdg-utils
```

Thanks to DeepMind, Werner Duvaud and Aurèle Hainaut