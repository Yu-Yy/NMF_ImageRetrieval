# NMF_ImageRetrieval
Applying augment NMF method to realize the image retrieval function: you can refer to our codes via [github repository](https://github.com/Yu-Yy/NMF_ImageRetrieval).

## Basic requirement
The codes are running under:
```
numpy
scikit-learn
scikit-image
opencv-python
pickle
csv
```
In order to install the gist package, please refer to this [link](https://github.com/Kalafinaian/python-img_gist_feature)

## Result

| Method     | Acc1     | Acc5     | Precision  |  Recall  | F1  |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: |
| NMF_(image)     | 0.0028    | 0.0056   | x | x    |  x    |
| NMF_(feature)     | 0.4275    | 0.5175     | 0.3048 | 0.2571    |  0.2640    |
| PCA_(image)     | 0.5650    | 0.6025     | 0.4429 | 0.3862    |  0.3930      |
| PCA_(feature)     | 0.4800     | 0.5350     | 0.3268 | 0.2829 | 0.2890 |
| Aug_NMF_(feature)    | 0.6125    | 0.6625     | 0.4620 | 0.4052  | 0.4317  |
| HRnet     | 0.6125    | 0.7200     | 0.4396 | 0.3897  | 0.3987  |
| HRnet+Transformer     | 0.5725     | 0.6225     | 0.4297 | 0.3708  | 0.3858  |


## Dataset download
You can download the dataset of `shopee-product-matching/` by this [link](https://cloud.tsinghua.edu.cn/f/5c7ba8c55e04478d86d9/)  and uncompress, put in this directory.

## Data preprocess
In our experiment, we divide the original dataset into training set, test set for training the deep nerual network. Besides, we 
extract a mini dataset (including 500 labels) for evaluate different methods. The evaluation method is that using the test set of the mini one as quiries and trainning set of that as gallaries. The divided datasets (for deep neural network trainning) file name is in `datasets/train_divide_eq.pkl`, `datasets/test_divide_eq.pkl`. Mini datasets file name is in `datasets/train_divide_mini500_eq.pkl`, `datasets/val_divide_mini500_eq.pkl`, `datasets/test_divide_mini500_eq.pkl`. <br>

## run the code
```
# Run the traditional nmf
python nmf_tradition.py
# Run the tradutional pca
python pca_test.py
# Run the augment nmf
python nmf_augment.py
```

## Cited
The deep learning result is tested by us in this [link](https://github.com/Yu-Yy/PR_project)


