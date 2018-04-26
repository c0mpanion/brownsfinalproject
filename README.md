## Installation
1. `git clone git@github.com:c0mpanion/brownsfinalproject.git`
2. `./install`
3. `python project`

## Usage
### Specify Thread Count
- `python project -threads 64`
- `python project -t 64`

## Notes
1. Make sure your `~/.bashrc` has a path to Cuda/nvcc `setenv PATH ${PATH}:/usr/local/cuda/bin` then run `source ~/.bashrc`