import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from utils.common_utils import model_performance
from utils.processor import create_edited_sentences


def get_initial_datasets():
    train_df = pd.read_csv('data/task-1/train.csv')
    test_df = pd.read_csv('data/task-1/dev.csv')

    training_data = train_df['original']
    training_edits = train_df['edit']
    training_grades = train_df['meanGrade']

    edited_training = pd.Series(create_edited_sentences(training_data, training_edits))
    training_dataset, validation_dataset, training_gradeset, validation_gradeset = train_test_split(edited_training,
                                                                                                    training_grades,
                                                                                                    test_size=0.2,
                                                                                                    random_state=42)

    vectorizer = CountVectorizer()
    training_bags_of_words = vectorizer.fit_transform(training_dataset)
    validation_bag_of_words = vectorizer.transform(validation_dataset)

    return training_bags_of_words, validation_bag_of_words, training_gradeset, validation_gradeset


def run_SVR_experiment():
    training_bags_of_words, validation_bag_of_words, training_gradeset, validation_gradeset = get_initial_datasets()

    model = SVR(kernel='rbf', C=0.1)
    model = model.fit(training_bags_of_words, training_gradeset)
    predictions = model.predict(validation_bag_of_words)

    test_mse, test_rmse, _ = model_performance(predictions, validation_gradeset)
    print(f'| Test Set MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} |')


def run_RandomForestRegressor_experiment():
    training_bags_of_words, validation_bag_of_words, training_gradeset, validation_gradeset = get_initial_datasets()

    model = RandomForestRegressor()
    model = model.fit(training_bags_of_words, training_gradeset)
    predictions = model.predict(validation_bag_of_words)

    test_mse, test_rmse, _ = model_performance(predictions, validation_gradeset)
    print(f'| Test Set MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} |')


def run_LinearRegression_experiment():
    training_bags_of_words, validation_bag_of_words, training_gradeset, validation_gradeset = get_initial_datasets()

    model = LinearRegression()
    model = model.fit(training_bags_of_words, training_gradeset)
    predictions = model.predict(validation_bag_of_words)

    test_mse, test_rmse, _ = model_performance(predictions, validation_gradeset)
    print(f'| Test Set MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} |')


def run_Pipeline_experiment():
    training_bags_of_words, validation_bag_of_words, training_gradeset, validation_gradeset = get_initial_datasets()

    model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
    model = model.fit(training_bags_of_words, training_gradeset)
    predictions = model.predict(validation_bag_of_words)

    test_mse, test_rmse, _ = model_performance(predictions, validation_gradeset)
    print(f'| Test Set MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} |')


def run_MultiLayerPerceptron_experiment():
    training_bags_of_words, validation_bag_of_words, training_gradeset, validation_gradeset = get_initial_datasets()

    model = MLPRegressor()
    model = model.fit(training_bags_of_words, training_gradeset)
    predictions = model.predict(validation_bag_of_words)

    test_mse, test_rmse, _ = model_performance(predictions, validation_gradeset)
    print(f'| Test Set MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} |')
