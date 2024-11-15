# SMS Text Classification with Gradio

This repository contains a Jupyter Notebook that builds an SMS text classification model using machine learning. The model predicts whether a given SMS message is "spam" or "not spam." Additionally, the notebook uses Gradio to create an interactive web interface, making it easy to test the model with custom input.

# Project Overview

## Goal
The primary goal of this project is to classify SMS messages as spam or not spam using a machine learning model, and provide an interactive UI using Gradio.

## Notebook Features
1. Data Preprocessing: Includes steps to clean and prepare the SMS text data for modeling.
2. Model Training: Trains a machine learning model to classify SMS messages.
3. Evaluation: Evaluates the model's performance on test data.
4. Gradio Interface: Deploys an interactive UI to test the SMS classifier.

## Getting Started

### Prerequisites
- Python (version >= 3.7)
- Jupyter Notebook or Google Colab
- Required Libraries: Install the necessary Python libraries using the following command:

``` python  
pip install pandas scikit-learn gradio

```
## Running the Notebook

### Clone the Repository:

``` python 
git clone https://github.com/Sami_Mangal/sms-text-classification
cd sms-text-classification
```
## Open the Notebook:
- Open gradio_sms_text_classification.ipynb in Jupyter Notebook or Google Colab.
- Run All Cells:
- Execute each cell in the notebook to preprocess the data, train the model, and launch the Gradio interface.

## Using the Gradio Interface

After running all cells in the notebook, a Gradio interface should appear, allowing you to enter SMS messages and see if the model classifies them as "spam" or "not spam."

## File Structure

gradio_sms_text_classification.ipynb: Main notebook containing the code for SMS text classification and Gradio UI setup.

### Model Description

The model uses a standard NLP pipeline with a classifier (e.g., Naive Bayes, Logistic Regression) to classify SMS messages as spam or not spam. Preprocessing steps include text cleaning, tokenization, and vectorization (e.g., TF-IDF).

## Example Usage

When you run the notebook, the Gradio interface will allow you to type in SMS messages. Here’s an example of how to use the interface:

Type a message like "Congratulations! You've won a $1000 Walmart gift card."
Click "Submit."
The model will display the prediction: spam.

## Future Enhancements

- Experiment with different NLP preprocessing techniques.
- Train using advanced classifiers like SVM or a neural network.
- Improve the Gradio interface for better user experience.

### Acknowledgments

Special thanks to the Gradio team for providing a simple UI for machine learning models.

