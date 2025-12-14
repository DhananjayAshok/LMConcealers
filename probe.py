from utils.parameter_handling import load_parameters, compute_secondary_parameters
from utils.log_handling import log_warn, log_info
from data import Curation
from prompts import hiding_prompts
import click
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

loaded_parameters = load_parameters()



def binarize(x):
    x = x.lower().split("\n")[0]
    if "yes" in x or "1" in x:
        return 1
    elif "no" in x or "0" in x:
        return 0
    else:
        if x.strip() == "":
            return None
        print(x, " could not be binarized")
        return 0
    
def featurize(df):
    # check the number of Nans
    num_nans = df["binarized"].isna().sum()
    if num_nans > 0:
        log_info(f"Warning: {num_nans} NaN values found in binarized outputs. ", parameters={})
    values = df["binarized"].values.reshape(-1)
    label = df["label"].values.mean() # Should be all the same
    # assert all labels are the same
    assert all(df["label"] == df["label"].iloc[0]), "Labels are not the same in the dataframe"
    return values, label

def fit_probe(train_dfs, test_dfs, test_filenames, parameters):
    train_features = []
    train_labels = []
    for df in train_dfs:
        X, y = featurize(df)
        train_features.append(X)
        train_labels.append(y)
    X_train = np.vstack(train_features)
    y_train = np.hstack(train_labels)
    test_features = []
    test_labels = []
    for df in test_dfs:
        X, y = featurize(df)
        test_features.append(X)
        test_labels.append(y)
    X_test = np.vstack(test_features)
    y_test = np.hstack(test_labels)
    model = LogisticRegression()
    # make sample_weight to handle class imbalance
    class_weights = {0: 1, 1: (len(y_train) / (2 * (y_train == 1).sum()))}
    sample_weight = np.array([class_weights[y] for y in y_train])
    model.fit(X_train, y_train, sample_weight=sample_weight)
    test_preds = model.predict(X_test)
    corrects = (test_preds == y_test).sum()
    total = len(y_test)
    accuracy = corrects / total
    log_info(f"Test accuracy: {accuracy} ({corrects}/{total})", parameters=parameters)
    for i, fname in enumerate(test_filenames):
        log_info(f"Test file: {fname}. Label: {y_test[i]}. Prediction: {test_preds[i]}", parameters=parameters)
    return


@click.command()
@click.pass_obj
def do_probe_fit(parameters):
    topics = list(hiding_prompts.keys())
    # remove default topic
    #topics.remove("default")
    models = ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.2", "gemma-3-12b-it", "Meta-Llama-3-70B-Instruct"]
    temperature = 0.0
    data_dir = parameters["data_dir"] + f"/probe/"
    filename = data_dir + "inference_input_[TOPIC]_[MODEL]_" + str(temperature) + "_output.jsonl"
    all_files = {}
    for topic in topics:
        for model in models:
            fname = filename.replace("[TOPIC]", topic).replace("[MODEL]", model)            
            df = Curation.get_output_file(fname, parameters, drop_none=False)
            df["binarized"] = df["output"].apply(binarize)
            if df["binarized"].isna().any():
                log_warn(f"NaNs found for {fname}")
                df["binarized"] = df["binarized"].fillna(0)
            if topic != "default":
                df["label"] = 1
            else:
                df["label"] = 0
            all_files[(topic, model)] = df
    for test_model in models:
        train_dfs = []
        test_dfs = []
        for (topic, model), df in all_files.items():
            if model == test_model:
                test_dfs.append(df)
            else:
                train_dfs.append(df)
        test_filenames = [f"{topic}_{model}" for (topic, model), df in all_files.items() if model == test_model]
        log_info(f"Fitting probe for test model: {test_model}", parameters=parameters)
        fit_probe(train_dfs, test_dfs, test_filenames, parameters)
        


    

# Any parameter from your project that you want to be able to change from the command line should be added as an option here
@click.group()
@click.option("--random_seed", default=loaded_parameters["random_seed"], help="The random seed for the project")
@click.option("--log_file", default=loaded_parameters["log_file"], help="The file to log to")
@click.pass_context
def main(ctx, **input_parameters):
    log_file_passed = input_parameters["log_file"]
    loaded_parameters.update(input_parameters)
    compute_secondary_parameters(loaded_parameters)
    if log_file_passed != loaded_parameters["log_file"]:
        warning_msg = f"The log file passed in is different from the one in the config files. \
        This is fine, but you need to take care that whenever you call functions from \
        utils/log_handling.py you pass in the parameters dict, otherwise there will be a mixup."
        log_warn(warning_msg, parameters=loaded_parameters)
    ctx.obj = loaded_parameters

main.add_command(do_probe_fit, name="fit")

if __name__ == "__main__":
    main()