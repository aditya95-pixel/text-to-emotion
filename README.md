## Emotion Classifier based on text analysis

### Overview
This repository contains a Python script for training and deploying an emotion classifier using machine learning techniques. The classifier is built using various libraries such as pandas, numpy, seaborn, and scikit-learn, along with NeatText for text preprocessing.

### Dependencies
- pandas
- numpy
- seaborn
- scikit-learn
- NeatText (install via `pip install neattext`)

### Dataset
The emotion dataset used (`emotion_dataset_raw.csv`) should be placed in the same directory as the script. This dataset contains raw text data labeled with different emotions.

### Usage
1. **Data Loading and Preprocessing:**
   - The dataset is loaded using pandas (`pd.read_csv`).
   - Initial data exploration is performed (`df.head()` and `df['Emotion'].value_counts()`).
   - Text preprocessing is done using NeatText functions to remove user handles and stopwords.

2. **Model Training:**
   - The dataset is split into training and testing sets using `train_test_split` from scikit-learn.
   - A text classification pipeline (`Pipeline` from scikit-learn) is created. It includes:
     - `CountVectorizer` for converting text into numerical features.
     - `LogisticRegression` as the classifier.
   - The pipeline is trained using `fit` on the training data.

3. **Model Evaluation:**
   - The trained model's performance is evaluated on the test set using accuracy score (`pipe_lr.score`).

4. **Prediction and Serialization:**
   - Example predictions are demonstrated with a sample text (`ex1`).
   - The model is serialized using `joblib.dump` to create `emotion_classifier_pipe_lr.pkl`.

5. **Deployment:**
   - To deploy the trained model:
     - Load the serialized model (`joblib.load`).
     - Use the loaded model to make predictions on new data.

### Files
- `emotion_dataset_raw.csv`: Raw dataset containing text and emotion labels.
- `model.py`: Python script for training the emotion classifier.

### Running the Script
To run the script:
```bash
python model.py
```
Ensure all dependencies are installed and the dataset (`emotion_dataset_raw.csv`) is in the same directory as the script.

### Notes
- This script assumes the dataset is formatted correctly and contains necessary columns (`Text` for text data and `Emotion` for labels).
- Adjust parameters (e.g., `test_size`, `random_state`, classifier options) as needed for your specific use case.


---


