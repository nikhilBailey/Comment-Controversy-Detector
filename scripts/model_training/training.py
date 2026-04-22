
class Model {
    # Fields: model_name, num_epochs, evaluate_every, optimization_function, (ml model itself), metrics (list of recorded metrics)
}

PERCENT_TEST_DATA = 0.1


def import_data() -> pd.DataFrame:
    #TODO: import the data from the csv file
    #TODO: return the data as a pandas dataframe


def test_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    #TODO: split the data into a training and testing set randomly based on PERCENT_TEST_DATA
    #TODO: return the training and testing sets as a tuple of pandas dataframes

def cross_validation_split(num_folds: int, training_data: pd.DataFrame) -> list[list[pd.DataFrame]]:
    #TODO: split all the training data into num_folds folds randomly
    #TODO: return the folds as a list of lists of pandas dataframes


def train_model(model: Model, folds: list[list[pd.DataFrame]], validation_fold: pd.DataFrame, evaluate_every: int) -> float:
    #TODO: train the model on the training data
    #TODO: return the metrics of the model every evaluate_every epochs

def evaluate_model(model: Model, validation_fold: pd.DataFrame) -> float:
    #TODO: evaluate the model on the validation set
    #TODO: return the accuracy, percision, recall, f1-score, pearson correlation, and spearman correlation of the model


def produce_model_training_visualizations(Model: Model, output_path: str) -> None: # metrics as a list of touples
    # TODO: produce charts for accuracy, percision, recall, f1-score, pearson correlation, and spearman correlation over time
    # Save these charts to a subfolder in visualizations with the timestamp and model name


def produce_final_model_evaluation_visualizations(models: list[Model], output_path: str) -> None:
    # Show the final perfomance of all the models using several visualizations


def produce_encoder_visualizations(Model: Model, output_path: str) -> None:
    # Show visualizations on the data learned by the encoder of the model


def produce_all_visualizations(models: list[Model], output_dir: str) -> None:
    # create a directory with the timestamp
    # create a directory for each set of visualizations
    # remember to have directory for specifically visualizations on only the top model
    # produce all the visualizations in their respective directories

if __name__ == "__main__":

    models = [
        Model()
        Model()
        Model()
        # about 8 models here
    ]
    #TODO: train all the models

    #TODO: produce final model evaluation visualization
    #TODO: produce encoder visualization for the top model
    #TODO: produce model training visualizations for each model

