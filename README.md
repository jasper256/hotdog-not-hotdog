# hotdog-not-hotdog
This Silicon Valley-inspired application uses machine learning to classify images as either containing or not containing a hotdog.

## About
The machine learning model used is a convolutional neural network built using Tensorflow/Keras. It is trained on RGB images 100 pixels x 100 pixels.
The web application is built using Streamlit and allows a user to upload a JPG image of up to 200MB. The user will then be notified whether or not the picture contains a hotdog.

## Using
After cloning the repository and installing all dependencies listed in the Requirements.txt file, download some training and testing data such as [this dataset from Kaggle](https://www.kaggle.com/yashvrdnjain/hotdognothotdog). Configure your environment variables such that `TRAIN_DATA_DIR` and `TEST_DATA_DIR` each map to a folder (for training and testing data respectively) containing two subfolders: `hotdog/` and `nothotdog/`. In each, `hotdog/` will contain images containing hotdogs and `nothotdog/` will contain images without. If you wish to modify this directory structure, you will have to make minor modifications to the paths in `src/clean_data.py`. After that, simply navigate to the `src/` folder and run
```
$ python clean_data.py
$ python train.py
```
If you wish, you can modify either or both of these files in an attempt to improve accuracy, reduce, loss, or decrease training time of the model.
Finally, run the app with
```
streamlit run hotdog_app.py
```

## Contributing
Contributions are welcome, simply fork the repository and then create a pull request. Please try to keep code clean and stay as consistent as possible with the existing standards/conventions. For major changes, please create an issue first and get approval (that is if you want your changes merged onto the original project).
