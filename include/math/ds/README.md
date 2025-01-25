## ds (Data science)-submodule
`MATH_EXTENSION=ds`

Used for machine learning and data science tasks


## Dependencies
In case you want to use `ImageDataSet` you need to add the compile flag
`MATH_IMAGE_PROCESSING=1` to your build process as well as having the following libraries
installed on your machine:
- ImageMagick/Magick++`>=7.0.10-45`  
Ubuntu: `sudo apt-get install libgraphicsmagick1-dev`  
Arch: `sudo pacman -S imagemagick`


For MacOS/Windows build add
```c++
InitializeMagick(*argv);
```
to your main to initialize `Magick++`  
**Note:** unnecessary for UNIX systems


## TOC

| File | Usage |
|:--- | --- |
|**Data Science** | |
|`DataSet.h` | Represents a dataset |
|`ImageDataSet.h` | Represents a dataset of image files |
|**Machine Learning** | |
|*Linear Classifiers*| |
|`NCC.h`| Nearest Centroid Classifier |
|*Non-linear Classifiers*| |
|`KNN.h` | K Nearest Neighbor Classifier |
|*Neuronal Networks* | |
|`Perceptron.h` | Perceptron classifier |
|`AdalineGD.h` | Adaline Neuron classifier trained using the gradient decent method |
|`AdalineSGD.h` | Adaline Neuron classifier trained using the statistical gradient decent method |
