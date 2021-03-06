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
After training, you can evaluate the reconstruction performance with the trained weights by the following code:
```
python test.py --target_midi '/path_to_your_data/JSB-Chorales-dataset/jsb-chorales-16th.pkl' --weights '/weights/weight.pth'
```
It will randomly pick 2 songs from the dataset and generate 3 outputs: the masked original content song, the generated song, and the style songs. Also it will display a heatmap visualization of the generated song.

### Lattice plot
After finish training, you can view the lattice plot with the trained weights using the following script:
```
python lattice.py --weights '/weights/weight.pth'
```
