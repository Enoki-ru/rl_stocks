### Description

This project is designed to explore the capabilities of `SL` + `RL` and their collaboration capabilities.
The project is an attempt to create an AI more advanced than humanity.
As we know, it is the `RL` models that are capable of learning, and, in fact, they are able to learn indefinitely, they can even become smarter than humans.

### Useful Links

- Author: [Sergei Voronin](https://github.com/Enoki-ru)

### Sources

All 
1. `data/raw/us_stocks` – [Stocks-Daily-Price (Hugging Face)](https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price/tree/main/data)


### Importing Data

- We are using `find data -type d -exec touch {}/.gitkeep \;` to create all `.gitkeep` files inside `/data/` folder
- To adding `.gitkeep` to repo, we also using command `git add -f data/**/.gitkeep`
- To work with the data, pre-upload all files from **Sources** to the `data/raw/<dataset_name>` folder.

### How to install venv

We are currently working on GPU on **macOS** and **Windows**, so:
- We added `tensorflow-metal` and `tensorflow-macos` if `sys_platform == 'darwin'` for **macOS**
- We added `tensorflow==2.15.0` for **Windows**

```bash
cd env
python3.11 -m pip install virtualenv
python3.11 -m venv ml_env
source ml_env/bin/activate
pip3.11 install -r reqs.req.txt
python -m ipykernel install --user --name=project-venv --display-name "ML-torch env"
```

#### How to change generate req.txt

We are using `pip-tools` to find versions stability

```bash
pip3.11 install pip-tools
pip-compile --resolver=backtracking -o req.txt req.in
```
