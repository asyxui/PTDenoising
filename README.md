# Path Tracing Denoiser

## How to setup Python
If you don't have Python installed already, download it from https://www.python.org/downloads/

1. Create a virtual environment in the root folder of the project.
```sh
python -m venv venv
```
2. Activate the virtual environment.
```sh
# Windows (cmd / powershell)
venv\Scripts\activate

# Mac/Linux (bash / zsh)
source venv/bin/activate
```
You will see (venv) appear in your terminal if you've done it right.

3. Install dependencies 
```sh
pip install -r requirements.txt
```

## Run Dataset creation script
1. Make sure you have completed the section [How to setup Python](#how-to-setup-python) and have activated the virtual environment.
2. Make sure you have Blender installed and added to path.
3. In your terminal, navigate to ./dataset
4. Start the database creation using this command:
```sh
blender -b -P .\blenderDataset.py
```
