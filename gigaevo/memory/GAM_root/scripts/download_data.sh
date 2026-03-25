## for locomo
mkdir -p data/locomo
cd data/locomo
wget https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
cd ../..

## for hotpotqa
mkdir -p data/hotpotqa
cd data/hotpotqa
wget https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_400.json
wget https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_1600.json
wget https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_6400.json
cd ../..

## for ruler
python download_data/download_ruler.py

## for narrativeqa
python download_data/download_narrativeqa.py
