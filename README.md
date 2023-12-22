## How to setup

```bash
# create the venv
python -m venv .env

# enter the venv
.env/bin/activate # mac/linux
.env/Scripts/activate.ps1 # windows (powershell)
```

Then, we install the dependencies.

To get PyTorch, go [here](https://pytorch.org/get-started/locally/), select the right OS and copy the `pip` command.

Then, to install gymnasium:

```bash
pip3 install gymnasium[classic-control]
```

Finally, to run the program:

```
python main.py
```