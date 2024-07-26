# ProtoPNet for multiple datasets

Fork of [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) to make it work with multiple datasets:
* Birds (like the original repo)
* StanfordCars
* StanfordDogs

Steps:
1. Run `img_crop.py` to convert the dataset to crop format.
2. Then `img_aug.py` to create augmented version.
3. Then you can run `main.py` to train.