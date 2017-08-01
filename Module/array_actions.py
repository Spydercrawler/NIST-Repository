# Some basic stuff:
import numpy as np
from collections import Iterable
import copy
# Sklearn classifiers:
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.svm import SVC
# Other Sklearn tools:
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils import shuffle
# Matplotlib stuff:
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_variable_array(independent_variable_arr,
                        dependent_variable_arr,
                        num_lines="all",
                        title=None,
                        axis="on",
                        show=False):
    """
    This method takes two arrays containing lists of values and graphs lines from the information in the arrays.
    :param independent_variable_arr: a two dimensional array containing lists of values for the x-axis.
    :param dependent_variable_arr: a two dimensional array containing lists of values for the y-axis
    :param num_lines: The amount of lines to be graphed.
    :param title: The title of the graph.
    :param axis: Whether or not the axis should be disabled or enabled. valid values are "on" and "off"
    :param show: Whether or not the method should call plt.show()
    """
    # These lines get the actual amount of lines to graph, since strings can be inputted in num_lines.
    # It will also raise an error if num_lines is not a valid string or int, so that is just an added bonus!
    new_num_lines = None
    if isinstance(num_lines, int):
        new_num_lines = num_lines
    elif isinstance(num_lines, basestring):
        new_num_lines = convert_number_string_to_integer(num_lines, len(independent_variable_arr))
    else:
        raise ValueError("num_points is not an int or a string!")
    # Before the method actually graphs lines, it checks to see if the amount of lines the user wants to graph
    # is bigger than the amount of lines that are possible to be graphed, just in case.
    if new_num_lines <= len(independent_variable_arr) and new_num_lines <= len(dependent_variable_arr):
        # The method plots the amount of lines specified by the method call.
        for i in range(new_num_lines):
            plt.plot(independent_variable_arr[i], dependent_variable_arr[i])
        # The method will add a title if the user specified a title.
        if title is not None:
                plt.title(title)
        # The method sets the axis to be what the user specified.
        plt.axis(axis)
        # If the show is set to True, the method shows the graph.
        if show:
            plt.show()
    else:
        raise ValueError("num_lines is bigger than the amount of lines that are possible to graph!")


def train_classifier(classifier_type='Random Forest', **data_arrays):
    """
    Accepts arrays of datasets created by ArrayDictionarySystem's get_dataset_variable_values method and
    returns a classifier that has been trained off of the data specified.
    :param classifier_type: The type of classifier to be trained.
    A classifier CLASS (not object) can be inputted,
    or a string can be inputted with the name of the classifiers;
    keep in mind this does not work for every classifier available, but a lot of classifiers are valid.
    :param data_arrays: keyword arguments with the key containing the prediction value and
    the value being an a array of datasets created by ArrayDictionarySystem's get_dataset_variable_values method
    :return: a classifier that has been trained off of the data specified.
    """
    clf = None
    if isinstance(classifier_type, basestring):
        # A giant dictionary of classifiers in case the user inputted a string for classifier_type.
        classifiers = {
            "randomforest": RandomForestClassifier(),
            "quadraticdiscriminantanalysis": QuadraticDiscriminantAnalysis(),
            "gaussiannb": GaussianNB(),
            "gaussiannaivebayes": GaussianNB(),
            "bernoullinb": BernoulliNB(),
            "bernoullinaivebayes": BernoulliNB(),
            "multinomialnb": MultinomialNB(),
            "multinomialnaivebayes": MultinomialNB(),
            "decisiontree": DecisionTreeClassifier(),
            "ridge": RidgeClassifier(),
            "sgd": SGDClassifier(),
            "stochasticgradientdescent": SGDClassifier(),
            "passiveaggressive": PassiveAggressiveClassifier(),
            "perceptron": Perceptron(),
            "svc": SVC()
        }
        # Now, because people can type things in in different ways,
        # we try to make this string system as fool-proof as possible!
        # First, we set all the characters to lower case:
        formatted_classifier_type = classifier_type.lower()
        # Then we remove all underscores and spaces:
        formatted_classifier_type = formatted_classifier_type.translate(None, " _")
        # Then we remove all instances of the word "classifier":
        formatted_classifier_type = formatted_classifier_type.replace("classifier", "")
        # Now, we check to see if they actually inputted a classifier we know! If they didn't, we raise a ValueError.
        if formatted_classifier_type in classifiers:
            clf = classifiers[formatted_classifier_type]
        else:
            raise ValueError("Classifier name inputted is not a supported classifier type!")
    else:
        # So what would happen if the individual inputted their own classifier type for a classifier?
        # Well, this covers that! Using the check_estimator method, we can tell if the inputted classifier is valid,
        # and raise an error if it isn't.
        try:
            check_estimator(classifier_type)
        except KeyboardInterrupt:
            raise KeyboardInterrupt("")
        except:
            raise ValueError("Estimator inputted is not a valid estimator, use sklearn's check_estimator method for more information!")
        else:
            clf = classifier_type()
    # Now we need to get our data in the correct format! First, we define two lists:
    # the training list which will contain the data,
    # and the target list which will contain what value each dataset is supposed to be.
    training_list = []
    target_list = []
    # Now we iterate through our keyword arguments, to add values to these two lists.
    for key, value in data_arrays.iteritems():
        # If the value at the current key is a numpy array, we just add all of it's values to the training list,
        # and it's key multiple times to the target list.
        if isinstance(value, np.ndarray):
            training_list.extend(value.tolist())
            target_list.extend([key] * len(value))
        # We do the same thing with the list.
        elif isinstance(value, list):
            training_list.extend(value)
            target_list.extend([key] * len(value))
        # If the value at the current key is another form of iterable, we convert it to a list and do the same thing.
        elif isinstance(value, Iterable):
            value_as_list = list(value)
            training_list.extend(value_as_list)
            target_list.extend([key] * len(value_as_list))
        # If the value is not any type of iterable, we raise a ValueError.
        else:
            raise ValueError("keyword argument has been inputted that is not an array or iterable")
    # Now we convert both lists to arrays, because they must be arrays to be put through the classifier.
    training_arr = np.array(training_list)
    target_arr = np.array(target_list)
    # To make sure no bias appears in the classifier, we shuffle both of the arrays in the same way.
    training_arr, target_arr = shuffle(training_arr, target_arr, random_state=0)
    # Finally, we fit the classifier and return it.
    clf.fit(training_arr, target_arr)
    return clf


def predict_data_with_classifier(trained_classifier, *data_arrays):
    """
    This takes a trained classifier and a variable amount of arrays of data created by
    ArrayDictionarySystem's get_dataset_variable_values method and returns the predictions of the classifier for the
    arrays of data, in a list.
    :param trained_classifier: A trained classifier.
    :param data_arrays: Arrays of data created by ArrayDictionarySystem's get_dataset_variable_values method.
    :return: predictions of the classifier for the arrays of data,
    in the format that the data_arrays arguments was inputted in.
    """
    if len(data_arrays) > 1:
        # Since I want this to be as open ended as possible,
        # I am trying to allow people to input multiple arrays of data.
        # Because of this, since more than one value in data_arrays was inputted,
        # we first need to go through all of the arrays of data and add them to a universal testing_data list.
        # Also, technically, they don't have to be arrays, but I would like them to be.
        testing_data = []
        for arg in data_arrays:
            if isinstance(arg, np.ndarray):
                testing_data.append(arg)
            elif isinstance(arg, list):
                testing_data.append(np.array(arg))
            elif isinstance(arg, Iterable):
                testing_data.append(np.array(list(arg)))
            else:
                raise ValueError("Argument inputted that is not an array, list, or other Iterable!")
        # Now, it just predicts and returns the predicted data.
        result_list = []
        # I prefer going through lists with numerical indexes, so I did that,
        # but this can easily be changed if it needs to be.
        for index in range(len(testing_data)):
            result_list.append(trained_classifier.predict(testing_data[index]))
        return result_list
    elif len(data_arrays) == 1:
        # Well, if we have a single argument inputted, we don't want to be returning a single value list,
        # so this is intended to stop that from happening. I won't comment anything else in this part because
        # it is basically a simplified version of the above code.
        if isinstance(data_arrays[0], np.ndarray):
            return trained_classifier.predict(data_arrays[0])
        elif isinstance(data_arrays[0], list):
            return trained_classifier.predict(np.array(data_arrays[0]))
        elif isinstance(data_arrays[0], Iterable):
            trained_classifier.predict(np.array(list(data_arrays[0])))
        else:
            raise ValueError("Argument inputted that is not an array, list, or other Iterable!")


def predict_data_with_known_type_with_classifier(trained_classifier, **data_arrays):
    """
    This method does the same thing as the predict_data_with_classifier method,
    except returns two lists: expected and predicted. expected contains the actual type of each dataset,
    and predicted contains the predicted type of each dataset.
    :param trained_classifier: a trained classifier
    :param data_arrays: Arrays of data created by ArrayDictionarySystem's get_dataset_variable_values method.
    :return: a tuple containing the expected array first and the predicted array second.
    """
    # Before I start explaining everything, I want to say that, for certain reasons, this method does not retain the
    # structure of data_arrays like the predict_data_with_classifier does. It simply returns two lists: expected,
    # and predicted, because I couldn't figure out how to do anything else. If somebody needs to get their original
    # data into a list of the same format, they can use the get_multiple_data_arrays_as_list method.
    #
    # Alright, so this following part is basically a copy-pasted version of code in the train_classifier method,
    # so if you do not understand this, I recommend you go there.
    test_list = []
    expected = []
    for key, value in data_arrays.iteritems():
        if isinstance(value, np.ndarray):
            test_list.extend(value.tolist())
            expected.extend([key] * len(value))
        elif isinstance(value, list):
            test_list.extend(value)
            expected.extend([key] * len(value))
        elif isinstance(value, Iterable):
            value_as_list = list(value)
            test_list.extend(value_as_list)
            expected.extend([key] * len(value_as_list))
        else:
            raise ValueError("keyword argument has been inputted that is not an array or iterable")
    # We now turn the two lists into arrays, and shuffle them:
    expected = np.array(expected)
    test_list = np.array(test_list)
    test_list, expected = shuffle(test_list, expected, random_state=0)
    # Next we use the classifier to predict the data.
    predicted = trained_classifier.predict(test_list)
    # Finally, we return both lists.
    return expected, predicted


def make_prediction_graph(trained_classifier,
                          x_axis,
                          data_array,
                          num_lines="all",
                          title=None,
                          axis="on",
                          show=False,
                          z_indexes=None):
    """
    Takes a trained classifier and a lot of information and
    draws a graph depicting lines that went through the classifier,
    with their color indicating what the classifier predicted they were.
    :param trained_classifier: A trained classifier to be used for predictions of the data specified.
    :param x_axis: some type of iterable object containing values to be used for
    the x values of points in datasets specified by the kwargs.
    :param data_array: An array containing all the datasets to be tested by the classifier.
    :param num_lines: the amount of lines to be plotted on the graph. Accepted values are any integer,
    the string "all" if you want all lines to be plotted,
    and the string "half" if you want half of the lines to be plotted.
    :param title: The title of the graph.
    :param axis: use this to specify whether you want the axes to be turned on or off.
    Accepted values are "on" and "off".
    :param show: A boolean specifiying whether or not you want plt.show() to be called by the function.
    :param z_indexes: a dictionary with keys being all of the kwarg keys,
    which contains z indexes for different predicted results.
    If not specified or set to None, the default z indexes will be used.
    """
    # First, right off the bat, we shuffle the data_array variable, and use our classifier to get predicted values.
    test_list = shuffle(data_array, random_state=0)
    predicted = trained_classifier.predict(test_list)
    # Next, we set some variables to be used later.
    color_types = {}
    graph_handles = []
    # Now, we get the actual amount of lines we need to plot.
    new_num_lines = None
    if isinstance(num_lines, int):
        new_num_lines = num_lines
    elif isinstance(num_lines, basestring):
        new_num_lines = convert_number_string_to_integer(num_lines, len(test_list))
    else:
        raise ValueError("num_lines must be an integer or a string!")
    # Now we start going through each line and what the classifier predicted for each one.
    for index, values, predicted_value in zip(range(len(test_list)), test_list, predicted):
        # If we have gone past the amount of lines we want to graph, we break out of the for loop.
        if index > new_num_lines:
            break
        # Now we check to see if the predicted value for this line has been plotted before,
        # and therefore already has a color.
        if predicted_value in color_types:
            # If it does, we check to see if z_indexes is not None.
            # If it isn't None, then we give it its assigned z index; otherwise, we leave it at the default.
            if z_indexes is not None:
                plt.plot(x_axis, values,
                         label=predicted_value,
                         color=color_types[predicted_value],
                         zorder=z_indexes[predicted_value])
            else:
                plt.plot(x_axis, values, label=predicted_value, color=color_types[predicted_value])
        else:
            # Right now, we have a line that has a predicted value that has not been plotted yet.
            # Because of this, first we have to plot it:
            plotted_stuff = None
            if z_indexes is not None:
                plotted_stuff = plt.plot(x_axis, values, label=predicted_value, zorder=z_indexes[predicted_value])
            else:
                plotted_stuff = plt.plot(x_axis, values, label=predicted_value)
            # Now, since matplotlib has automatically assigned this line a color,
            # we add the color to the color_types dictionary,
            # with it's key being the predicted value for the current line.
            color_types[predicted_value] = plotted_stuff[0].get_color()
            # We also add a handle for this color, for the legend later on.
            graph_handles.append(
                mpatches.Patch(color=color_types[predicted_value],
                               label=predicted_value))
    # Now we create a legend showing all of the colors of the lines and all of their respective values.
    # The legend gets added to the side, though I may add functionality to edit it's location later.
    plt.legend(handles=graph_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # If the user has specified a title, it will be added:
    if title is not None:
        plt.title(title)
    # The axis setting is applied to the plot:
    plt.axis(axis)
    # If the user set show to true, the plot will be shown:
    if show:
        plt.show()


def convert_number_string_to_integer(number_string, max_values):
    if number_string.isdigit():
        return int(number_string)
    string_dict = {
        "all": 1,
        "half": 2,
        "third": 3,
        "quarter": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    if number_string in string_dict:
        return max_values//string_dict[number_string]
    else:
        raise ValueError("String inputted does not contain an integer or a valid indication of percentage!")


def get_multiple_data_arrays_as_list(*data_arrays):
    """
    Since the predict_data_with_known_type_with_classifier method only returns two lists, this method exists to get
    a set of multiple data arrays into the same list format as the predict_data_with_known_type_with_classifier method.
    :param data_arrays:
    :return:
    """
    data_array_combined_list = []
    for arg in data_arrays:
        if isinstance(arg, np.ndarray):
            data_array_combined_list.append(arg)
        elif isinstance(arg, list):
            data_array_combined_list.append(np.array(arg))
        elif isinstance(arg, Iterable):
            data_array_combined_list.append(np.array(list(arg)))
        else:
            raise ValueError("Argument inputted that is not an array, list, or other Iterable!")
    return data_array_combined_list


def compare_classifiers_in_graph(good_datasets,
                                 randomness_amplitude_range,
                                 title=None,
                                 axis="on",
                                 show=False,
                                 progress_printing=False,
                                 **classifier_types):
    classifier_results = {}
    final_range = []
    for key in classifier_types.keys():
        classifier_results[key] = []
    for amplitude in randomness_amplitude_range:
        bad_datasets = copy.deepcopy(good_datasets)
        for index in range(len(bad_datasets)):
            bad_datasets[index] = (np.array(bad_datasets[index]) +
                                   np.random.normal(0, amplitude, len(bad_datasets[index]))).tolist()
        new_good_datasets = copy.deepcopy(good_datasets)
        training_data = np.concatenate((new_good_datasets, bad_datasets))
        for index in range(len(training_data)):
            for value in range(len(training_data[index])):
                training_data[index][value] = abs(training_data[index][value])
        target_data = ["good"] * len(good_datasets)
        target_data.extend(["bad"] * len(bad_datasets))
        training_data, target_data = shuffle(training_data, target_data, random_state = 0)
        n_sets = len(training_data)
        for key in classifier_types.keys():
            clf = classifier_types[key]()
            clf.fit(training_data[:n_sets//2], target_data[:n_sets//2])
            expected = target_data[n_sets//2:]
            predicted = clf.predict(training_data[n_sets//2:])
            num_correct = 0
            for index in range(len(predicted)):
                if predicted[index] == expected[index]:
                    num_correct += 1
            classifier_results[key].append((num_correct/float(len(predicted)))*100.0)
        final_range.append(amplitude)
        if progress_printing:
            print "Finished Amplitude!"
    graph_handles = []
    for key in classifier_results.keys():
        graph_handles.extend(
            plt.plot(final_range, classifier_results[key], label=key)
        )
    print graph_handles
    plt.legend(handles=graph_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if title is not None:
        plt.title(title)
    plt.axis(axis)
    if show:
        plt.show()
