# CSC412-Deep Beethoven

### Dataset
You can get the data from https://github.com/czhuang/JSB-Chorales-dataset

### Training
To train the model with default hyperparameters
```
# Script to start training the model
python train.py --data 'path to data directory/JSB-Chorales-dataset/jsb-chorales-16th.pkl' --weights_dir './weights/'
```

### Testing
After training, you can evaluate the reconstruction performance by the following code:
```
python test.py --target_midi '/content/JSB-Chorales-dataset/jsb-chorales-16th.pkl' --style_midi '/content/JSB-Chorales-dataset/jsb-chorales-16th.pkl'
```
It will randomly pick 3 songs from the dataset and generate 4 outputs: the masked original song, the generated song, and the two style songs.

### Lattice plot
After finish training, you can view the lattice plot with the following script:
```
python lattice.py
```
