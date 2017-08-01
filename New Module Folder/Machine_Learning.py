# -----------------------------------------------------------------------------
# Name:        Machine Learning
# Purpose:     Functions and a class for loading, formatting and interpolating data, as well as generating data,
#              training classifiers off of data, and performing tests.
# Author:      John Bass
# Created:     7/31/2017
# -----------------------------------------------------------------------------
""" Machine_Learning is a module containing a class and functions for loading,
formatting, and interpolating data, as well as generating fake data to be used,
training classifiers off of data, and performing tests on data.

Requirements
------------
+ [copy](https://docs.python.org/2/library/copy.html)
+ [random](https://docs.python.org/2/library/random.html)
+ [collections](https://docs.python.org/2/library/collections.html)
+ [numpy](https://docs.scipy.org/doc/)
+ [scipy](https://docs.scipy.org/doc/)
+ [pandas](http://pandas.pydata.org/pandas-docs/stable/)
+ [statsmodels](http://www.statsmodels.org/stable/index.html)
+ [matplotlib](https://matplotlib.org/)
+ [scikit-learn](http://scikit-learn.org/stable/)

Examples
---------------
+ [Module Introduction](HTML_Examples/Module_Introduction.html)
+ [Fake Data Generation](HTML_Examples/Fake_Data_Creation.html)
+ [Classifier Comparison](HTML_Examples/Classifier_Comparisons.html)
"""
# Standard Modules:
import copy
from collections import Iterable
import random
# Non-Standard Modules:
try:
    import numpy as np
except:
    print("The module numpy either was not found or had an error"
          "Please put it on the python path, or resolve the error")
    raise
try:
    import pandas as pd
except:
    print("The module numpy either was not found or had an error"
          "Please put it on the python path, or resolve the error")
    raise
try:
    from scipy import interpolate
except:
    print("The module scipy.interpolate either was not found or had an error"
          "Please put it on the python path, or resolve the error")
    raise
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except:
    print("The module statsmodels.nonparametric.smoothers_lowess.lowess either was not found or had an error"
          "Please put it on the python path, or resolve the error")
    raise
try:
    from sklearn.utils import shuffle
except:
    print("The module sklearn.utils.shuffle either was not found or had an error"
          "Please put it on the python path, or resolve the error")
    raise
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except:
    print("The module matplotlib was not found."
          "Please put it on the python path.")
    raise


class DictionarySystem(object):
    """DictionarySystem is a class that contains a nested dictionary that has data that has been split by
    unique values of one or more variables. It is used to format and interpolate data,
    as well as generate entirely new data."""
    # Functions:
    def __init__(self, basedict):
        """
        To initialize a DictionarySystem,
        a nested dictionary must be inputted with the same format that a DictionarySystem uses.

        An easy way to create a DictionarySystem is to use any of the following 4 functions:

        * split_table_by_column
        * split_table_by_columns
        * split_csv_by_column
        * split_csv_by_columns
        """
        # Raises an error if basedict is not a dictionary.
        if not isinstance(basedict, dict):
            raise ValueError("basedict must be a dictionary!")
        elif isinstance(basedict, dict):
            # Checks the validity of basedict. If basedict does not have the correct format,
            # or has value types that are not DictionarySystems, dictionaries,
            # or pandas Dataframes, raises a ValueError.
            if self.__check_dictionary_validity(basedict):
                self.dictionary = basedict
                # Converts all sub-dictionaries into DictionarySystems to
                # allow all of the recursive methods to work.
                self.__convert_dicts_to_systems()
            else:
                raise ValueError("Basedict is not in a valid format, or has invalid value types!")
        # This is just in case something goes dreadfully wrong, but hopefully it will never be called.
        else:
            raise ValueError("Alright, I don't know what you even did, but something is wrong with basedict.")

    def __getitem__(self, key):
        return self.dictionary[key]

    def __setitem__(self, key, value):
        # This is made to only accept DictionarySystems, dictionaries, and pandas DataFrames,
        # or else it will throw an error. I am sure you can figure this method out.
        if isinstance(value, (DictionarySystem, pd.DataFrame)):
            self.dictionary[key] = value
        elif isinstance(value, dict):
            self.dictionary[key] = DictionarySystem(value)
        else:
            raise ValueError("Value set is not a dictionary or pandas dataFrame!")

    def __len__(self):
        return len(self.dictionary)

    def __str__(self):
        return "DictionarySystem("+str(self.dictionary)+")"

    def __deepcopy__(self):
        return DictionarySystem(copy.deepcopy(self.dictionary))

    def __delitem__(self, key):
        del self.dictionary[key]

    # Old Dictionary Methods:
    def keys(self):
        """This returns the keys of the DictionarySystem's dictionary."""
        return self.dictionary.keys()

    def get(self, key, default=None):
        """
        Returns a value for a given key inside the DictionarySystem's dictionary.

        **key:** the key to be searched in the dictionary.

        **default:** The value to be returned in case the key specified does not exist.

        **returns:** the value for the given key in side the FDictionarySystem's dictionary.
        """
        return self.dictionary.get(key, default)

    def pop(self, key, default=None):
        """
        Pops the value at the given key from the DictionarySystem's dictionary and returns it.

        **key:** The key to pop the value from

        **default:** If the given key does not exist within the DictionarySystem's dictionary,
        this value will be returned.

        **returns:** The value that was just popped from the given key.
        """
        return self.dictionary.pop(self, key, default=default)

    def items(self):
        """This returns the key,value pairs of the DictionarySystem's dictionary in tuples."""
        return self.dictionary.items()

    def values(self):
        """This returns a list of all the values in the DictionarySystem's dictionary."""
        return self.dictionary.values()

    def iteritems(self):
        """Returns the iterator returned by the iteritems method of the DictionarySystem's dictionary."""
        return self.dictionary.iteritems()

    def iterkeys(self):
        """Returns the iterator returned by the iterkeys method of the DictionarySystem's dictionary."""
        return self.dictionary.iterkeys()

    def itervalues(self):
        """Returns the iterator returned by the itervalues method of the DictionarySystem's dictionary."""
        return self.dictionary.itervalues()

    # Table Methods:
    def split_by_column(self, column):
        """
        Splits the dictionary system's tables by a single column,
        making the location where the table was a dictionary containing unique column values in the column as keys,
        with each key containing a table.

        **column:** The column to split the dictionary systems by.
        """
        for key in self.keys():
            # If the value at that key is a DictionarySystem it calls split_by_column on that.
            if isinstance(self[key], DictionarySystem):
                self[key].split_by_column(column)
            # If the value at that key is a pandas DataFrame it tries to split it:
            elif isinstance(self[key], pd.DataFrame):
                # If the column specified is not in the DataFrame, the method raises a KeyError.
                if column in self[key] is False:
                    raise KeyError("Column Specified is not in a table!")
                unique_vals = self[key][column].unique()
                new_dict = {}
                # The method goes through all of the unique values in the column in the table,
                # creates new DataFrames containing only the unique value specified, and adds them to a dictionary.
                for val in unique_vals:
                    df = self[key][self[key][column] == val]
                    df = df.reset_index(drop=True)
                    new_dict[val] = df
                # The dictionary is then set to be the new value of the key.
                self[key] = DictionarySystem(new_dict)
            # The method will raise an error if the DictionarySystem contains
            # values other than DictionarySystems and pandas DataFrames.
            else:
                raise ValueError("DictionarySystem contains values other than DictionarySystems and tables!")

    def split_by_columns(self, *columns):
        """
        Does the same thing as the split_by_column method, except splits by multiple columns in order instead of one.

        **columns:** The names of each column to split the tables by.
        """
        # If no columns are specified, the method raises a ValueError
        if len(columns) == 0:
            raise ValueError("At least one column value must be specified!")
        # Since we now know that columns has some values in it, the method flattens the list of columns.
        # I added this so users can input complex nested list systems and stuff as arguments for this method.
        columns_flattened = list(self.__better_flatten(columns))
        # Now, the method goes through the columns_flattened list and checks if each one of them is a string.
        # If one of them is not a string, it clearly isn't a column name, so the methos raises a ValueError.
        for val in columns_flattened:
            if not isinstance(val, basestring):
                raise ValueError("Value other than Iterable or String specified as a column!")
        # Jeez, I just realized that like 90% of this entire method is error checking.
        # Anyway, this now checks if the length of columns_flattened is 0, and if it is, throws an error.
        # I added this because otherwise people could put in a lot of lists and no string column names
        # and the method would be okay with it.
        if len(columns_flattened) == 0:
            raise ValueError("At least one column value must be specified!")
        # finally, after all that error checking, the method goes through the columns_flattened list in order, and
        # calls split_by_column with each column.
        for column in columns_flattened:
            self.split_by_column(column)

    def remove_column_duplicates(self, column):
        """
        Removes all rows that have a duplicate value in a specific column
        except one in every table in the DictionarySystem,
        so that no duplicates in the said column will remain in the DictionarySystem.

        The way this removes columns is by going through each DataFrame in the DictionarySystem, and,
        for each DataFrame, a new DataFrame is created containing the first rows containing unique values in the initial
        DataFrame. The old DataFrame is then overwritten.

        **column:** The column used to remove duplicate values.
        """
        for key in self.keys():
            # If the value at the current key is a DictionarySystem, it calls remove_column_duplicates on that.
            if isinstance(self[key], DictionarySystem):
                self[key].remove_column_duplicates(column)
            # otherwise, if the value at the current key is a pandas DataFrame, it starts removing column duplicates.
            elif isinstance(self[key], pd.DataFrame):
                # It sets the variable new_dataframe to none since I didn't want the variable to be local to
                # the for loop and I couldn't figure out any other way to do that.
                new_dataframe = None
                # The method goes through all the unique values of the column specified.
                unique_vals = self[key][column].unique()
                for val in unique_vals:
                    # Inside the for loop, the method creates a new DataFrame containing all the rows where the value
                    # in the specified column is the same as the for loop's unique value. It then selects the first
                    # row in that DataFrame and adds it to the new_dataframe variable. The result at the end of the
                    # for loop is a new dataframe where duplicate values in the specified column have been removed.
                    df = self[key][self[key][column] == val]
                    if new_dataframe is None:
                        new_dataframe = df.iloc[[0]]
                    else:
                        new_dataframe = pd.concat([new_dataframe, df.iloc[[0]]])
                # Finally the method just resets the index of this new DataFrame and sets the value at the current key
                # to be the new DataFrame with duplicate values removed, replacing the old DataFrame.
                self[key] = new_dataframe.reset_index(drop=True)
            # The method will raise an error if the DictionarySystem contains
            # values other than DictionarySystems and pandas DataFrames.
            else:
                raise ValueError("DictionarySystem has a value that isn't a DictionarySystem or Pandas DataFrame!")

    def remove_short_tables(self, row_count):
        """
        Removes all tables in the DictionarySystem that are below a certain row count.

        **row_count:** The row count required for a table to stay in the DictionarySystem.
        """
        for key in self.keys():
            # If the value at the current key is a DictionarySystem, it calls remove_short_tables on that.
            # If the DictionarySystem has no tables left in it after that, the method deletes the current key.
            if isinstance(self[key], DictionarySystem):
                self[key].remove_short_tables(row_count)
                if self[key] is None or len(self[key]) <= 0:
                    del self[key]
            # If the value at the current key is a pandas DataFrame and it's row count is
            # less than the specified minimum row count, the method deletes the current key.
            elif isinstance(self[key], pd.DataFrame):
                if self[key].shape[0] < row_count:
                    del self[key]
            # If the value at the current key is not a DictionarySystem or pandas DataFrame,
            # the method raises a ValueError.
            else:
                raise ValueError("DictionarySystem has a value that isn't a DictionarySystem or Pandas DataFrame!")

    def keep_only_certain_columns(self, *columns):
        """
        Removes all columns in every table except the ones specified.
        This is generally used to remove irrelevant data from tables.

        **columns:** The names of the columns to keep in every table.
        """
        # If no columns are specified, the method raises a ValueError
        if len(columns) == 0:
            raise ValueError("At least one column value must be specified!")
        # Since we now know that columns has some values in it, the method flattens the list of columns.
        # I added this so users can input complex nested list systems and stuff as arguments for this method.
        columns_flattened = list(self.__better_flatten(columns))
        # Now, the method goes through the columns_flattened list and checks if each one of them is a string.
        # If one of them is not a string, it clearly isn't a column name, so the methos raises a ValueError.
        for val in columns_flattened:
            if not isinstance(val, basestring):
                raise ValueError("Value other than Iterable or String specified as a column!")
        # The method now checks if the length of columns_flattened is 0, and if it is, throws an error.
        # I added this because otherwise people could put in a lot of lists and no string column names
        # and the method would be okay with it.
        if len(columns_flattened) == 0:
            raise ValueError("At least one column value must be specified!")
        # Now, most of the error checking is done, so the method starts
        # going through the keys in the DictionarySystem.
        for key in self.keys():
            # If the value at the current key is a DictionarySystem,
            # the keep_only_certain_columns method is called on that.
            if isinstance(self[key], DictionarySystem):
                self[key].keep_only_certain_columns(columns_flattened)
            # If the value a the current key is a DataFrame, the method sets the value at the current key to a new
            # DataFrame that only has the columns specified.
            elif isinstance(self[key], pd.DataFrame):
                self[key] = self[key][columns_flattened]
            # If the value at the current key is not a DictionarySystem or pandas DataFrame,
            # the method will raise an error.
            else:
                raise ValueError(
                    "DictionarySystem has a value that isn't a DictionarySystem or Pandas DataFrame!")

    # Array Methods:
    def interpolate_data(self,
                         num_points,
                         independent_variable,
                         dependent_variables,
                         interpolation_kind="cubic"):
        """
        Interpolates independent and dependent variables in datasets within the DictionarySystem to have the amount
        of points specified. Unfortunately, columns that are not the independent or dependent variables will be removed,
        since they are not being interpolated and the dataframe has to stay rectangular.

        The independent variable must not have repeat values, and must be increasing. If these two requirements are not
        fulfilled, the method will fail.

        **num_points:** The amount of points the dependent variables will be interpolated to.

        **independent_variable:** The name of the independent variable of the dataset.
        All values of this variable must be increasing, and no duplicate values can exist.

        **dependent_variables:** An iterable containing the names of the dependent variables of the dataset.

        **interpolation_kind:** The kind of interpolation to be used. This will be fed into scipy's interp1d method.
        """
        # Gets all of the nested dictionaries within the DictionarySystem.
        # This basically just gets the entire nested dictionary structure without any DictionarySystems in it.
        nested_dictionaries = self.__get_nested_dictionaries()
        # This converts all dataframes at the end of the nested dictionary structure to dictionaries
        # containing arrays representing columns.
        self.__nested_dictionary_dataframes_to_arrays(nested_dictionaries)
        # Now that our data is in a format that is easy to work with, we actually interpolate our data.
        self.__nested_array_dictionary_interpolation(nested_dictionaries,
                                                     num_points,
                                                     independent_variable,
                                                     dependent_variables,
                                                     interpolation_kind=interpolation_kind)
        # Now that our data is interpolated, we convert all of the dictionaries containing
        # arrays representing columns back to dataframes.
        self.__nested_dictionary_arrays_to_dataframes(nested_dictionaries)
        # We replace our object's dictionary with the newly interpolated one.
        self.dictionary = nested_dictionaries
        # Now we simply convert all sub-dictionaries to DictionarySystems
        # to get everything back into the correct format.
        self.__convert_dicts_to_systems()

    def make_fake_data_system_noise(self,
                                    independent_var,
                                    dependent_vars,
                                    num_datasets,
                                    location,
                                    randomness_amplitudes):
        """
        Creates a new DictionarySystem containing automatically generated datasets created by adding random noise to
        the dependent variables of real datasets.

        **independent_var:** The name of the independent variable of the datasets.
        This will not have random noise added to it.

        **dependent_vars:** An Iterable containing the names of the dependent variables of the datasets.
        These will have random noise added to their values to generate new datasets.

        **num_datasets:** The amount of datasets to create

        **location:** The location to generate fake data at.

        **randomness_amplitudes** An iterable containing the random noise amplitudes for the random noise to be
        added to the values for each dependent variable.

        **returns:** A new DictionarySystem containing automatically generated datasets.
        """
        # Gets all of the nested dictionaries within the DictionarySystem.
        # This basically just gets the entire nested dictionary structure without any DictionarySystems in it.
        nested_dictionaries = self.__get_nested_dictionaries()
        # This converts all dataframes at the end of the nested dictionary structure to dictionaries
        # containing arrays representing columns.
        self.__nested_dictionary_dataframes_to_arrays(nested_dictionaries)
        # Now that our data is in a format that is somewhat easy to work with, we actually create the fake data.
        # I had to use a seperate method here since my method of creating fake data used recursion.
        new_dictionaries = self.__nested_array_fake_noise_data_creation(nested_dictionaries,
                                                                        independent_var,
                                                                        dependent_vars,
                                                                        num_datasets,
                                                                        location,
                                                                        randomness_amplitudes)
        # Now that we have our fake data, we convert all the dictonaries of arrays representing columns to dataframes.
        self.__nested_dictionary_arrays_to_dataframes(new_dictionaries)
        # Finally we convert this dictionary into a DictionarySystem and return it.
        return DictionarySystem(new_dictionaries)

    def make_fake_data_system_slope(self,
                                    independent_var,
                                    dependent_vars,
                                    num_datasets,
                                    location,
                                    starting_noises,
                                    slope_deviations,
                                    smoothing_fracs):
        """
        Creates a new DictionarySystem containing automatically generated datasets created by adding random noise to the
        first points of already existing datasets then adding random noise to the slopes between points of existing
        datasets and creating new points based off of those. After all this is done, a smoothing filter is applied to
        make the generated lines less jagged.

        **independent_var:** The name of the independent variable of the datasets.
        This will not be modified during fake data creation.

        **dependent_vars:** An Iterable containing the names of the dependent variables of the datasets.
        Values of these variables in existing datasets will have modifications applied to them to generate new datasets.

        **num_datasets:** The amount of datasets to be generated.

        **location:** The location to get datasets to be used to generate new datasets.

        **starting_noises:** An Iterable containing the random noise amplitudes for the random noise to be added to
        each dependent variable in existing datasets to generate new datasets.

        **slope_deviations:** An Iterable containing the random noise amplitudes for the random noise to be added to
        the slopes of the lines for each dependent variable to generate new datasets.

        **smoothing_fracs:** An Iterable containing the smoothing fracs for the lowess filter to apply to the
        generated lines for each dependent variable.

        **returns:** A new DictionarySystem containing automatically generated datasets.
        """
        # Gets all of the nested dictionaries within the DictionarySystem.
        # This basically just gets the entire nested dictionary structure without any DictionarySystems in it.
        nested_dictionaries = self.__get_nested_dictionaries()
        # This converts all dataframes at the end of the nested dictionary structure to dictionaries
        # containing arrays representing columns.
        self.__nested_dictionary_dataframes_to_arrays(nested_dictionaries)
        # Now that our data is in a format that is somewhat easy to work with, we actually create the fake data.
        # I had to use a seperate method here since my method of creating fake data used recursion.
        new_dictionaries = self.__nested_array_fake_slope_data_creation(nested_dictionaries,
                                                                        independent_var,
                                                                        dependent_vars,
                                                                        num_datasets,
                                                                        location,
                                                                        starting_noises,
                                                                        slope_deviations,
                                                                        smoothing_fracs)
        # Now that we have our fake data, we convert all the dictonaries of arrays representing columns to dataframes.
        self.__nested_dictionary_arrays_to_dataframes(new_dictionaries)
        # Finally we convert this dictionary into a DictionarySystem and return it.
        return DictionarySystem(new_dictionaries)

    def get_dataset_variable_values(self, dataset_variable, location=None):
        """
        This gets all of the values for a single variable, and returns them in an array containing lists containing the
        datapoints for the variable for a single dataset. This is used by a lot of methods, and in general is a format
        that is somewhat easy to work with.

        **dataset_variable:** The name of the variable to get all of the values for.

        **location:** The location to get the variable values from. If it is none, all of the values for the variable
        will be gotten.

        **returns:** All of the values of a single variable in an array, with the values being seperated into lists
        containing the values for a single dataset.
        """
        nested_dictionaries = self.__get_nested_dictionaries()
        self.__nested_dictionary_dataframes_to_arrays(nested_dictionaries)
        dataset_variable_values = self.__nested_dictionary_get_dataset_variable_values(nested_dictionaries,
                                                                                       dataset_variable,
                                                                                       location)
        return dataset_variable_values

    # Array Helper Methods:
    def __nested_dictionary_get_dataset_variable_values(self,
                                                        dictionary,
                                                        dataset_variable,
                                                        location=None):
        """
        Helper method used by the get_dataset_variable_values method that actually gets the values of the variables for
        each dataset and outputs them in the right format. This method uses recursion so it had to be seperate from the
        other method.

        **dictionary:** The dictionary to get the values of the variables for each dataset from.

        **dataset_variable:** The name of the variable to get values from.

        **location:** The location to get the data from. If location is none, it will get dataset variable values
        from the entire dictionary.

        **returns** An array containing lists containing the values for a specific variable in each dataset,
        at the specified location.
        """
        # If no location was inputted, it simply gets all of the dataset
        # variable values in every dataset and returns an array of those.
        if location is None:
            # This list will contain the list of values of the dataset variable for every dataset in
            # the DictionarySystem.
            variable_set_list = []
            for key in dictionary.keys():
                # If the value at the first key in the current dictionary is an array,
                # it adds the array whose key is the dataset variable name to variable_set_list
                if isinstance(dictionary[key].values()[0], np.ndarray):
                    variable_set_list.append(dictionary[key][dataset_variable].tolist())
                # if the current location is an DictionarySystem that contains DictionarySystems representing
                # individual datasets, it goes through them calls get_dataset_variable_values to each one,
                # then adds them all to variable_set_list.
                elif isinstance(dictionary[key].values()[0].values()[0], np.ndarray):
                    key_results = self.__nested_dictionary_get_dataset_variable_values(dictionary[key],
                                                                                       dataset_variable,
                                                                                       location=[]).tolist()
                    if isinstance(key_results[0], list):
                        variable_set_list.extend(key_results)
                    else:
                        variable_set_list.append(key_results)
                # if the current location is higher up in the nesting, the method calls get_dataset_variable_values on
                # the DictionarySystem at the current location and adds the results to variable_set_list.
                else:
                    key_results = self.__nested_dictionary_get_dataset_variable_values(dictionary[key],
                                                                                       dataset_variable,
                                                                                       location=None).tolist()
                    if isinstance(key_results[0], list):
                        variable_set_list.extend(key_results)
                    else:
                        variable_set_list.append(key_results)
            # Finally, variable_set_list is returned as an array.
            return np.array(variable_set_list)
        # Since location can also be a string, if it is, it checks to see if location is an empty string.
        # If location is an empty string, it calls get_dataset_variable_values with location being an empty tuple.
        # If location is not an empty string, it calls get_dataset_variable_values with location being a
        # single value tuple with the only value being the string that was just passed to location.
        elif isinstance(location, basestring):
            if len(location) == 0:
                return self.__nested_dictionary_get_dataset_variable_values(dictionary,
                                                                            dataset_variable,
                                                                            location=[]).tolist()
            return self.__nested_dictionary_get_dataset_variable_values(dictionary,
                                                                        dataset_variable,
                                                                        location=[location]).tolist()
        # If location is an iterable, but not a string, then that means we are looking at some sort of path.
        elif isinstance(location, Iterable) and not isinstance(location, basestring):
            # If the length of location is 0, then that means that the current location is the location to
            # get the dataset variable's values from, so we go through the DictionarySystem at the current location
            # and add all of the values of the dataset variable to a list, then return it as an array.
            if len(location) == 0:
                variable_set_list = []
                for key in dictionary.keys():
                    variable_set_list.append(dictionary[key][dataset_variable].tolist())
                return np.array(variable_set_list)
            # If the length of location is 1 then we can do the same thing as when location is 0 with a bit of tweaking.
            elif len(location) == 1:
                current_location = location[0]
                variable_set_list = []
                for key in dictionary[current_location].keys():
                    variable_set_list.append(dictionary[current_location][key][dataset_variable].tolist())
                return np.array(variable_set_list)
            # If the length of location is greater than one then we just use recursion to our advantage.
            else:
                return self.__nested_dictionary_get_dataset_variable_values(dictionary[location[0]],
                                                                            dataset_variable,
                                                                            location=location[1:]).tolist()
        # If location is not None, a string, or an Iterable, we raise a ValueError.
        else:
            raise ValueError("location must be an Iterable!")

    def __nested_array_fake_slope_data_creation(self,
                                                dictionary,
                                                independent_var,
                                                dependent_vars,
                                                num_datasets,
                                                location,
                                                starting_noises,
                                                slope_deviations,
                                                smoothing_fracs):
        """
        The method used by the make_fake_data_system_slope method to generate fake data. This method is seperate from
        that method because this method requires the use of recursion.

        **dictionary:** The dictionary that will be used as a reference for the fake data generation.

        **independent_var:** The name of the independent variable of the datasets.
        This will not be modified during fake data creation.

        **dependent_vars:** An Iterable containing the names of the dependent variables of the datasets.
        Values of these variables in existing datasets will have modifications applied to them to generate new datasets.

        **num_datasets:** The amount of datasets to be generated.

        **location:** The location to get datasets to be used to generate new datasets.

        **starting_noises:** An Iterable containing the random noise amplitudes for the random noise to be added to
        each dependent variable in existing datasets to generate new datasets.

        **slope_deviations:** An Iterable containing the random noise amplitudes for the random noise to be added to
        the slopes of the lines for each dependent variable to generate new datasets.

        **smoothing_fracs:** An Iterable containing the smoothing fracs for the lowess filter to apply to the
        generated lines for each dependent variable.

        **returns:** A nested dictionary containing the newly generated data.
        """
        if isinstance(location, Iterable) and len(location) > 0 and not isinstance(location, basestring):
            if len(location) == 1:
                current_location = location[0]
                new_data_dict = {}
                for fake_set_num in range(num_datasets):
                    # Copying a random dataset in our already existing datasets,
                    # which will be modified to become fake data.
                    new_set_values = copy.deepcopy(dictionary[current_location]
                                                   [random.choice(dictionary[current_location].keys())])
                    # This will go through each dependent variable's datapoints and modify them.
                    for dep_var_index in range(len(dependent_vars)):
                        # This is the array of the old points from an existing dataset.
                        dep_var_values = new_set_values[dependent_vars[dep_var_index]]
                        new_dep_var_points = []
                        # This adds a random value to the starting value of dep_var_values,
                        # creating the first point in our fake dataset.
                        starting_value = dep_var_values[0] + np.random.normal(0, starting_noises[dep_var_index])
                        new_dep_var_points.append(starting_value)
                        for set_index in range(len(dep_var_values)-1):
                            # We need to modify the slope, so we first have to get the slope.
                            # To do this, we first have to get the change in the x value:
                            delta_x = new_set_values[independent_var][set_index+1] -\
                                      new_set_values[independent_var][set_index]
                            # We also need to get the change in the y value:
                            delta_y = dep_var_values[set_index+1]-dep_var_values[set_index]
                            # Now we simply calculate the slope:
                            segment_slope = delta_y / delta_x
                            # We then add a random value to the slope,
                            # and use our new slope to calculate the next point.
                            segment_slope += np.random.normal(0, slope_deviations[dep_var_index])
                            new_point_y_val = new_dep_var_points[set_index] + (delta_x * segment_slope)
                            new_dep_var_points.append(new_point_y_val)
                        # Now that we have all of our points, we apply a smoothing filter to our points so that
                        # our curve will not have as many sharp edges.
                        smoothed_dep_var_points = lowess(new_dep_var_points,
                                                         new_set_values[independent_var],
                                                         is_sorted=True,
                                                         frac=smoothing_fracs[dep_var_index],
                                                         it=0)[:,1]
                        new_set_values[dependent_vars[dep_var_index]] = smoothed_dep_var_points
                    new_data_dict["Fake Dataset " + str(fake_set_num)] = new_set_values
                return new_data_dict
            else:
                # if the length of location is more than one, we simply use recursion to narrow down the location:
                return self.__nested_array_fake_slope_data_creation(dictionary[location[0]],
                                                                    independent_var,
                                                                    dependent_vars,
                                                                    num_datasets,
                                                                    location[1:],
                                                                    starting_noises,
                                                                    slope_deviations,
                                                                    smoothing_fracs)
        elif isinstance(location, basestring):
            return self.__nested_array_fake_slope_data_creation(dictionary,
                                                                independent_var,
                                                                dependent_vars,
                                                                num_datasets,
                                                                [location],
                                                                starting_noises,
                                                                slope_deviations,
                                                                smoothing_fracs)
        else:
            raise ValueError("Location must be an iterable or a string!")

    def __nested_array_fake_noise_data_creation(self,
                                                dictionary,
                                                independent_var,
                                                dependent_vars,
                                                num_datasets,
                                                location,
                                                randomness_amplitudes):
        """
        The method used by the make_fake_data_system_noise method to generate fake data. This method is seperate from
        that method because this method requires the use of recursion.

        **dictionary:** The dictionary that will be used as a reference for the fake data generation.

        **independent_var:** The name of the independent variable of the datasets.
        This will not have random noise added to it.

        **dependent_vars:** An Iterable containing the names of the dependent variables of the datasets.
        These will have random noise added to their values to generate new datasets.

        **num_datasets:** The amount of datasets to create.

        **location:** The location to generate fake data at.

        **randomness_amplitudes:** An iterable containing the random noise amplitudes for the random noise to be
        added to the values for each dependent variable.

        **returns:** A nested dictionary containing the newly generated data.
        """
        if isinstance(location, Iterable) and len(location) > 0 and not isinstance(location, basestring):
            # If it's length is one, that means that means that all of the values in the current dictionary
            # are SUPPOSED to be dictionaries representing singular datasets.
            if len(location) == 1:
                # the current_location variable is just there so I don't have to keep writing location[0] a ton.
                current_location = location[0]
                # new_data_dict will contain all of the new, fake datasets.
                new_data_dict = {}
                for i in range(num_datasets):
                    # We need a deep copy of one of the datasets so that
                    # when we add random noise it won't affect the original dataset:
                    new_set_values = copy.deepcopy(dictionary[current_location]
                                                   [random.choice(dictionary[current_location].keys())])
                    # Now we add random noise to every single dependent variable:
                    for dependent_var, amp in zip(dependent_vars, randomness_amplitudes):
                        num_values_in_dataset = len(new_set_values[dependent_var])
                        new_set_values[dependent_var] += np.random.normal(0, amp, num_values_in_dataset)
                    # We now have a fake dataset, so we add it to new_data_dict, which contains all of our fake datasets
                    new_data_dict["Fake Dataset " + str(i)] = new_set_values
                # ...and now we just return new_data_dict!
                return new_data_dict
            else:
                # if the length of location is more than one, we simply use recursion to narrow down the location:
                return self.__nested_array_fake_noise_data_creation(dictionary[location[0]],
                                                                    independent_var,
                                                                    dependent_vars,
                                                                    num_datasets,
                                                                    location[1:],
                                                                    randomness_amplitudes)
        # If location is a string, the same method is called, but location is now a single element list,
        # so that The above code can do it's magic.
        elif isinstance(location, basestring):
            return self.__nested_array_fake_noise_data_creation(dictionary,
                                                                independent_var,
                                                                dependent_vars,
                                                                num_datasets,
                                                                [location],
                                                                randomness_amplitudes)
        # If location is not an iterable or a string, a ValueError is raised.
        else:
            raise ValueError("Location must be an iterable or a string!")

    def __nested_dictionary_dataframes_to_arrays(self, dictionary):
        """
        This method converts a nested dictionary with dataframes at the end of the nesting to a nested dictionary with
        dictionaries containing arrays representing columns at the end of the nesting.

        **dictionary:** The dictionary to modify.
        """
        # If dictionary is not a dict, an error is raised.
        if not isinstance(dictionary, dict):
            raise ValueError("parameter 'dictionary' is not a dict, it is a " + str(type(dictionary)))
        for key in dictionary.keys():
            # If the value at the current key is a dataframe,
            # it gets converted into a dictionary containing lists representing columns.
            if isinstance(dictionary[key], pd.DataFrame):
                dictionary[key] = dictionary[key].to_dict(orient='list')
                # All of the lists are now converted to arrays.
                for subkey in dictionary[key].keys():
                    dictionary[key][subkey] = np.array(dictionary[key][subkey])
            # If the value at the current key is a dict, it gets the method called on it.
            elif isinstance(dictionary[key], dict):
                self.__nested_dictionary_dataframes_to_arrays(dictionary[key])

    def __nested_dictionary_arrays_to_dataframes(self, dictionary):
        """
        This method converts a nested dictionary with dictionaries containing arrays representing columns at the end of
        the nesting to a nested dictionary with dataframes at the end of the nesting.

        **dictionary: The dictionary to modify.
        """
        # If dictionary is not a dict, an error is raised.
        if not isinstance(dictionary, dict):
            raise ValueError("parameter 'dictionary' is not a dict, it is a " + str(type(dictionary)))
        for key in dictionary.keys():
            # If the value at the current key is a dictionary containing arrays representing columns,
            # It gets converted to a dataframe.
            if isinstance(dictionary[key].values()[0], np.ndarray):
                dictionary[key] = pd.DataFrame.from_dict(dictionary[key])
            # If the value at the current key is a dict, it gets the method called on it.
            elif isinstance(dictionary[key].values()[0], dict):
                self.__nested_dictionary_arrays_to_dataframes(dictionary[key])

    def __nested_array_dictionary_interpolation(self,
                                                dictionary,
                                                num_points,
                                                independent_variable,
                                                dependent_variables,
                                                interpolation_kind="cubic"):
        """
        This is the method that actually interpolates the data in a nested dictionary, and is only called by
        the interpolate_data method. The reason why this method and the interpolate_data method are separate methods is
        because this method requires recursion.

        **dictionary:** A nested dictionary which will have its datasets interpolated.

        **num_points:** The amount of points the dependent variables will be interpolated to.

        **independent_variable:** The name of the independent variable of the dataset.
        All values of this variable must be increasing, and no duplicate values can exist.

        **dependent_variables:** An iterable containing the names of the dependent variables of the dataset.

        **interpolation_kind:** The kind of interpolation to be used. This will be fed into scipy's interp1d method.
        """
        for key in dictionary.keys():
            if isinstance(dictionary[key].values()[0].values()[0], np.ndarray):
                maximum_of_mins = None
                minimum_of_maxes = None
                for dataset_key in dictionary[key].keys():
                    if independent_variable not in dictionary[key][dataset_key].keys():
                        raise KeyError("Independent variable specified is not a key in the DictionarySystem!")
                    # All of this here is simple code to find these maxes of minimums and minimums of maxes:
                    if maximum_of_mins is None:
                        maximum_of_mins = dictionary[key][dataset_key][independent_variable].min()
                    elif dictionary[key][dataset_key][independent_variable].min() > maximum_of_mins:
                        maximum_of_mins = dictionary[key][dataset_key][independent_variable].min()
                    if minimum_of_maxes is None:
                        minimum_of_maxes = dictionary[key][dataset_key][independent_variable].max()
                    elif dictionary[key][dataset_key][independent_variable].max() < minimum_of_maxes:
                        minimum_of_maxes = dictionary[key][dataset_key][independent_variable].max()
                # Now that we have the bounds for our interpolation, we need to make our interpolation functions:
                for dataset_key in dictionary[key].keys():
                    # First we need to get the values for the independent and
                    # dependent variables in this specific dataset:
                    independent_var_values = dictionary[key][dataset_key][independent_variable].tolist()
                    dependent_var_values = [
                        dictionary[key][dataset_key][variable_name].tolist() for variable_name in dependent_variables]
                    # This list will hold the interpolation functions soon:
                    dependent_var_interpolation_functions = []
                    for dependent_var_list in dependent_var_values:
                        # Now, we simply use scipy's interp1d function to get our interpolation functions:
                        var_function = interpolate.interp1d(independent_var_values,
                                                            dependent_var_list,
                                                            interpolation_kind)
                        dependent_var_interpolation_functions.append(var_function)
                    # The new independent variable values should be uniform, so we just use np.linspace:
                    new_independent_var_values = np.linspace(maximum_of_mins, minimum_of_maxes, num_points)
                    # This list will be used to store our dependent variable values
                    # once we use the interpolation functions we just made:
                    new_dependent_var_values = []
                    try:
                        # Now we go through our interpolation functions and
                        # get new values for all of our dependent variables:
                        for interp_function in dependent_var_interpolation_functions:
                            new_dependent_var_values.append(interp_function(new_independent_var_values))
                        # We make a dictionary out of these new values using variable names as keys:
                        interpolated_dictionary = dict(zip(dependent_variables, new_dependent_var_values))
                        interpolated_dictionary[independent_variable] = new_independent_var_values
                        # Finally, we replace our old DictionarySystem for this dataset with
                        # our newly interpolated one. By the way, we are inputting a dictionary, but the __setitem__
                        # method will convert it into an DictionarySystem, so we do not have to worry about that.
                        dictionary[key][dataset_key] = interpolated_dictionary
                    # Sometimes interpolation fails because some dataset's bounds are outside of the
                    # interpolation range we just got, so this is here to just remove all the datasets that do that.
                    except ValueError:
                        del dictionary[key][dataset_key]
                        continue
            elif isinstance(dictionary[key].values()[0].values()[0], dict):
                self.__nested_array_dictionary_interpolation(dictionary[key],
                                                             num_points,
                                                             independent_variable,
                                                             dependent_variables,
                                                             interpolation_kind=interpolation_kind)
            else:
                raise ValueError("Something went wrong, change this error message later")

    # Other Helper Methods:
    def __get_nested_dictionaries(self):
        """
        This gets all of the dictionaries in the nested DictionarySystems
        and returns them as a nested dictionary structure.
        """
        # This will soon contain all of the nested dictionaries in the DictionarySystem.
        nested_dict = {}
        # This basically goes through the DictionarySystem, calling the same method on sub-DictionarySystems and
        # appending their results to the dictionary, as well as adding any non-DictionarySystem values to the dictionary
        for key in self.keys():
            if isinstance(self[key], DictionarySystem):
                nested_dict[key] = self[key].__get_nested_dictionaries()
            else:
                nested_dict[key] = self[key]
        # Finally, our nested dictionary is just returned. Due to recursion, a full nested dictionary structure will
        # Eventually be outputted.
        return nested_dict

    def __check_dictionary_validity(self, dictionary):
        """
        Checks if a dictionary has a valid format and valid variable types to become a DictionarySystem.

        **dictionary:** The dictionary to check.

        **returns:** True or False depending on whether a dictionary has a valid format and valid variable types.
        """
        # If the dictionary variable inputted is not a dictionary, returns False.
        if isinstance(dictionary, (DictionarySystem, dict)):
            for key in dictionary.keys():
                # If any value in the dictionary is not a DictionarySystem, dictionary, or
                # pandas DataFrame, returns False.
                if isinstance(dictionary[key], (DictionarySystem, dict)):
                    # If the __check_dictionary_validity method fails on any sub dictionaries, returns False.
                    if not self.__check_dictionary_validity(dictionary[key]):
                        return False
                elif isinstance(dictionary[key], pd.DataFrame):
                    continue
                else:
                    return False
        else:
            return False
        # If it never returned False, returns True.
        return True

    def __convert_dicts_to_systems(self):
        """
        Converts all sub-dictionaries in the DictionarySystem's main dictionary into DictionarySystems.
        """
        # Literally all this does is if any value in the DictionarySystem is a dictionary, it gets turned into a
        # DictionarySystem.
        for key in self.keys():
            if isinstance(self[key], dict):
                self[key] = DictionarySystem(self[key])

    def __better_flatten(self, iterable_object):
        """Some flattening method I found on stackoverflow, which I a few methods
        to make the flattening of my columns arguments work."""
        for value in iterable_object:
            if isinstance(value, Iterable) and not isinstance(value, basestring):
                for newvalue in self.__better_flatten(value):
                    yield newvalue
            else:
                yield value


def split_table_by_column(table, column):
    """
    Takes a pandas DataFrame and splits it by column values into a DictionarySystem.

    **table:** The Dataframe to split into a DictionarySystem.

    **column:** The column to split the pandas DataFrame by.

    **returns:** A DictionarySystem created by splitting the pandas DataFrame inputted by the column inputted
    """
    table_dictionary = {}
    # Goes through every unique value in the specified column of the the table inputted
    unique_column_vals = table[column].unique()
    for val in unique_column_vals:
        # Adds to the dictionary a new table with every value in the
        # specified column being the current unique value from the for loop.
        table_dictionary[val] = table[table[column] == val]
    return DictionarySystem(table_dictionary)


def split_table_by_columns(table, *columns):
    """
    Takes a pandas DataFrame and splits it by all of the columns specified into a DictionarySystem

    **table:** The Dataframe to split into a DictionarySystem.

    **columns:** The columns to split the pandas DataFrame by.

    **returns:** A DictionarySystem created by splitting the pandas DataFrame inputted by the columns inputted
    """
    # Raises a ValueError if no columns are inputted:
    if len(columns) == 0:
        raise ValueError("At least one column must be specified!")
    # Splits the table by the first column:
    dictionary_system = split_table_by_column(table, columns[0])
    # If more than one column was specified, split the DictionarySystem by the extra columns as well.
    if len(columns) > 1:
        dictionary_system.split_by_columns(columns[1:])
    return dictionary_system


def split_csv_by_column(path, column):
    """
    Takes a path to a csv file, loads it as a pandas dataframe, splits it by a column,
    and returns the resulting DictionarySystem.

    **path:** The path to a csv file.

    **column:** The column to split the pandas dataframe by.

    **returns:** A DictionarySystem that is the result of splitting the
    table contained in the csv file specified by the column specified.
    """
    pandas_dataframe = pd.read_csv(path)
    return split_table_by_column(pandas_dataframe, column)


def split_csv_by_columns(path, *columns):
    """
    Takes a path to a csv file, loads it as a pandas dataframe, splits it by multiple columns,
    and returns the resulting DictionarySystem.

    **path:** The path to a csv file.

    **columns:** The columns to split the pandas dataframe by.

    **returns:** A DictionarySystem that is the result of splitting the
    table contained in the csv file specified by the columns specified.
    """
    pandas_dataframe = pd.read_csv(path)
    return split_table_by_columns(pandas_dataframe, *columns)


# Actions for value Arrays:
def plot_variable_array(independent_variable_arr,
                        dependent_variable_arr,
                        num_lines="all",
                        plot_axis=None,
                        **plot_kwargs):
    """
    This function takes two arrays containing lists of values and graphs lines from the information in the arrays.

    **independent_variable_arr:** a two dimensional array containing lists of values for the x-axis.

    **dependent_variable_arr:** a two dimensional array containing lists of values for the y-axis.

    **num_lines:** The amount of lines to be graphed.

    **plot_axis:** Optional argument, where a user can specify an axis variable to do matplotlib methods.

    **plot_kwargs:** keyword arguments to add to the matplotlib plot() method.
    Many will not be accepted, but most stylistic arguments will be.
    """
    if plot_axis is None:
        plot_axis = plt
    # These are the arguments that are going to be put into plot_axis.plot that can be changed.
    final_plot_args = {
        "lw": None,
        "alpha": None,
        "figure": None,
        "ls": None,
        "marker": None,
        "mec": None,
        "mew": None,
        "mfc": None,
        "ms": None,
        "markevery": None,
        "solid_capstyle": None,
        "solid_joinstyle": None,
        "color": None
    }
    # This goes and modifies arguments from the default values of "None" to their respective values in plot_args.
    for key in plot_kwargs:
        if key in final_plot_args:
            final_plot_args[key] = plot_kwargs[key]
    # These lines get the actual amount of lines to graph, since strings can be inputted in num_lines.
    # It will also raise an error if num_lines is not a valid string or int, so that is just an added bonus!
    new_num_lines = None
    if isinstance(num_lines, int):
        new_num_lines = num_lines
    elif isinstance(num_lines, basestring):
        new_num_lines = convert_number_string_to_integer(num_lines, len(independent_variable_arr))
    else:
        raise ValueError("num_points is not an int or a string!")
    # Before the function actually graphs lines, it checks to see if the amount of lines the user wants to graph
    # is bigger than the amount of lines that are possible to be graphed, just in case.
    if new_num_lines <= len(independent_variable_arr) and new_num_lines <= len(dependent_variable_arr):
        # The function plots the amount of lines specified by the function call.
        for i in range(new_num_lines):
            plot_axis.plot(independent_variable_arr[i], dependent_variable_arr[i], **final_plot_args)
    else:
        raise ValueError("num_lines is bigger than the amount of lines that are possible to graph!")


def train_classifier(classifier_type='Random Forest', **data_dictionary):
    """
    Accepts arrays of datasets created by DictionarySystem's get_dataset_variable_values method and
    returns a classifier that has been trained off of the data specified.

    **classifier_type:** The type of classifier to be trained.
    A classifier CLASS (not object) can be inputted,
    or a string can be inputted with the name of the classifiers;
    keep in mind this does not work for every classifier available, but a lot of classifiers are valid.

    **data_dictionary:** keyword arguments with the keys containing the prediction values and
    the values being an arrays of datasets created by DictionarySystem's get_dataset_variable_values method

    **returns:** a classifier that has been trained off of the data specified.
    """
    clf = None
    if isinstance(classifier_type, basestring):
        # I am importing inside the function since these will only be used in the specific circumstance that
        # a user inputted a string for classifier_type.
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, Perceptron
        from sklearn.svm import SVC
        # A giant dictionary of classifiers in case the user inputted a string for classifier_type:
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
        # Well, this covers that! Using the check_estimator function, we can tell if the inputted classifier is valid,
        # and raise an error if it isn't.
        # Keep in mind I am importing this here because if the user never inputs a classifier's class for
        # classifier_type, this import would never be used.
        from sklearn.utils.estimator_checks import check_estimator
        try:
            check_estimator(classifier_type)
        except KeyboardInterrupt:
            raise KeyboardInterrupt("")
        except:
            raise ValueError("Estimator inputted is not a valid estimator, " +
                             "use sklearn's check_estimator function for more information!")
        else:
            clf = classifier_type()
    # Now we need to get our data in the correct format! First, we define two lists:
    # the training list which will contain the data,
    # and the target list which will contain what value each dataset is supposed to be.
    training_list = []
    target_list = []
    # Now we iterate through our keyword arguments, to add values to these two lists.
    for key, value in data_dictionary.iteritems():
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
    DictionarySystem's get_dataset_variable_values method and returns the predictions of the classifier for the
    arrays of data, in a list.

    **trained_classifier:** A trained classifier.

    **data_arrays:** Arrays of data created by DictionarySystem's get_dataset_variable_values method.

    **returns:** predictions of the classifier for the arrays of data,
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


def predict_data_with_known_type_with_classifier(trained_classifier, **data_dictionary):
    """
    This function does the same thing as the predict_data_with_classifier function,
    except returns two lists: expected and predicted. expected contains the actual type of each dataset,
    and predicted contains the predicted type of each dataset.

    **trained_classifier:** a trained classifier

    **data_dictionary:** keyword arguments with the keys containing the prediction values and
    the values being an arrays of datasets created by DictionarySystem's get_dataset_variable_values method

    **returns:** a tuple containing the expected array first and the predicted array second.
    """
    # Before I start explaining everything, I want to say that, for certain reasons, this function does not retain the
    # structure of data_dictionary like the predict_data_with_classifier does. It simply returns two lists: expected,
    # and predicted, because I couldn't figure out how to do anything else. If somebody needs to get their original
    # data into a list of the same format, they can use the get_multiple_data_arrays_as_list function.
    #
    # Alright, so this following part is basically a copy-pasted version of code in the train_classifier function,
    # so if you do not understand this, I recommend you go there.
    test_list = []
    expected = []
    for key, value in data_dictionary.iteritems():
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
                          plot_axis=None,
                          z_indexes=None,
                          **plot_kwargs):
    """
    Takes a trained classifier and a lot of information and
    draws a graph depicting lines that went through the classifier,
    with their color indicating what the classifier predicted they were.

    **trained_classifier:** A trained classifier to be used for predictions of the data specified.

    **x_axis:** some type of iterable object containing values to be used for
    the x values of points in datasets specified by the kwargs.

    **data_array:** An array containing all the datasets to be tested by the classifier.

    **num_lines:** the amount of lines to be plotted on the graph. Accepted values are any integer,
    the string "all" if you want all lines to be plotted,
    and the string "half" if you want half of the lines to be plotted.

    **plot_axis:** Optional argument, where a user can specify an axis variable to do matplotlib methods.

    **z_indexes:** a dictionary with keys being all of the kwarg keys,
    which contains z indexes for different predicted results.
    If not specified or set to None, the default z indexes will be used.

    **plot_kwargs:** keyword arguments to add to the matplotlib plot() method.
    Many will not be accepted, but most stylistic arguments will be.
    """
    if plot_axis is None:
        plot_axis = plt
    # These are the arguments that are going to be put into plot_axis.plot that can be changed.
    final_plot_args = {
        "lw": None,
        "alpha": None,
        "figure": None,
        "ls": None,
        "marker": None,
        "mec": None,
        "mew": None,
        "mfc": None,
        "ms": None,
        "markevery": None,
        "solid_capstyle": None,
        "solid_joinstyle": None,
    }
    # This goes and modifies arguments from the default values of "None" to their respective values in plot_args.
    for key in plot_kwargs:
        if key in final_plot_args:
            final_plot_args[key] = plot_kwargs[key]
    # Now, we shuffle the data_array variable, and use our classifier to get predicted values.
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
                plot_axis.plot(x_axis, values,
                               label=predicted_value,
                               color=color_types[predicted_value],
                               zorder=z_indexes[predicted_value],
                               **final_plot_args)
            else:
                plot_axis.plot(x_axis, values,
                               label=predicted_value,
                               color=color_types[predicted_value],
                               **final_plot_args)
        else:
            # Right now, we have a line that has a predicted value that has not been plotted yet.
            # Because of this, first we have to plot it:
            plotted_stuff = None
            if z_indexes is not None:
                plotted_stuff = plot_axis.plot(x_axis, values,
                                               label=predicted_value,
                                               zorder=z_indexes[predicted_value],
                                               **final_plot_args)
            else:
                plotted_stuff = plt.plot(x_axis, values,
                                         label=predicted_value,
                                         **final_plot_args)
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
    plot_axis.legend(handles=graph_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def convert_number_string_to_integer(number_string, max_values):
    """
    This is a small function which is used in some other functions which,
    if number_string is an integer within a string, returns that integer,
    and if number_string is an indication of percentage, like "tenth" or "seventh" it returns that amount of max_values.
    For instance, if "third" is inputted for number_string, and max_values is thirty, 10 would be returned.

    **number_string:** The string containing the integer or indication of percentage.

    **max_values:** The maximum amount of values in a dataset or something,
    this is used for strings that indicate percentage.

    **returns:** The integer the number string contains or the integer calculated by the percentage number_string contains.
    """
    # If number_string contains only a number, it returns that number.
    if number_string.isdigit():
        return int(number_string)
    # These are the valid indications of percentage to be used.
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
    # If number_string is one of the valid indications of percentage, it returns the number that is that percentage.
    if number_string in string_dict:
        return max_values//string_dict[number_string]
    else:
        raise ValueError("String inputted does not contain an integer or a valid indication of percentage!")


def get_multiple_data_arrays_as_list(*data_arrays):
    """
    Since the predict_data_with_known_type_with_classifier function only returns two lists, this function exists to get
    a set of multiple data arrays into the same list format as the predict_data_with_known_type_with_classifier function.

    **data_arrays:** The arrays to turn into a list.

    **returns:** the combined list of all of the data arrays.
    """
    data_array_combined_list = []
    # This for loop just goes through the data_arrays dictionary and adds all the iterables in it to
    # data_array_combined_list as numpy arrays.
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


def get_classifier_comparision_results(good_datasets,
                                       randomness_amplitude_range,
                                       classifiers):
    """
    Takes an array of 'good' datasets, as well as an array of random noise amplitudes to test and classifier types
    as keyword arguments, and returns data to be graphed in the graph_comparison_results.
    These two functions together will make a graph showing the percentage of guesses that were correct at
    different random noise amplitudes for different classifiers. Users can use this to compare the performance
    of different classifiers. The reason why these functions are separate is because since this process takes a large
    amount of time, so if the functions are separated, it becomes a lot easier to quickly modify a plot to one's liking
    if the two functions are in an ipython notebook or something similar.

    **good_datasets:** An array of good datasets for a single variable.

    **randomness_amplitude_range:** An array all of the random noise amplitudes to use for "bad" data in tests.

    **classifiers:** A dictionary containing the classifier's CLASSES, not classifier objects, to test.
    Each key must be the name of each classifier.

    **returns:** Data to be used by the graph_comparison_results.
    """
    # This will contain the results at each point for each classifier.
    classifier_results = {}
    # This will contain all the amplitudes later.
    final_range = []
    # Each key of classifier_results will be the name of a classifier, and each value will be a list of floats,
    # each one detailling the percentage of predictions the classifier got correct at a specific point.
    for key in classifiers.keys():
        classifier_results[key] = []
    for amplitude in randomness_amplitude_range:
        # Now we generate an array of bad datasets, from the good ones.
        # We have to do this manually, since this isn't an DictionarySystem and we don't have a function for it.
        bad_datasets = copy.deepcopy(good_datasets)
        for index in range(len(bad_datasets)):
            bad_datasets[index] = (np.array(bad_datasets[index]) +
                                   np.random.normal(0, amplitude, len(bad_datasets[index]))).tolist()
        new_good_datasets = copy.deepcopy(good_datasets)
        # Although the following variable is called "training_data", we will also be using it for testing.
        # One half of it will be used to train each classifier, the other half will be used to test each classifier.
        training_data = np.concatenate((new_good_datasets, bad_datasets))
        # We need to make all of the values positive, since some classifiers
        # apparently don't work well with negative values.
        for index in range(len(training_data)):
            for value in range(len(training_data[index])):
                training_data[index][value] = abs(training_data[index][value])
        # Our target data will simply be "good" and "bad". At some point, I should probably add a
        # feature to be able to put in your own data to compare classifiers in non-binary classification,
        # but right now I am too tired.
        target_data = ["good"] * len(good_datasets)
        target_data.extend(["bad"] * len(bad_datasets))
        # To remove bias, we shuffle both the training and target datasets.
        training_data, target_data = shuffle(training_data, target_data, random_state=0)
        # This is the number of datasets, I am just defining this to make it easier in the next part.
        n_sets = len(training_data)
        for key in classifiers.keys():
            # We have to initialize the current classifier type here, and set it to a variable.
            clf = classifiers[key]()
            # Now we train it with half of the data.
            clf.fit(training_data[:n_sets // 2], target_data[:n_sets // 2])
            # This is what a classifier would predict if it got its predictions 100% correct.
            expected = target_data[n_sets // 2:]
            # This is what the classifier actually predicted.
            predicted = clf.predict(training_data[n_sets // 2:])
            # Now, we just add up how many predictions the classifier got right:
            num_correct = 0
            for index in range(len(predicted)):
                if predicted[index] == expected[index]:
                    num_correct += 1
            # We now use this number to calculate the percentage correct for this classifier
            # at this random noise amplitude, then add it to the classifier's results list.
            classifier_results[key].append((num_correct / float(len(predicted))) * 100.0)
        # We add the current amplitude to final_range.
        final_range.append(amplitude)
    # Now we just return a list containing classifier_results and final_range.
    return [classifier_results, final_range]


def graph_comparison_results(comparison_results,
                             plot_axis=None,
                             **plot_kwargs):
    """
    This takes in the data that was generated by the get_classifier_comparison_results function, and plots it into a
    graph comparing classifier results. The reason why these functions are separate is because the
    get_classifier_comparison_results function takes a lot of time, so if the functions are separated, it becomes a lot
    easier to quickly modify a plot to one's liking if
    the two functionss are in an ipython notebook or something similar.

    **comparison_results:** The results of the get_classifier_comparison_results function.

    **plot_axis:** Optional argument, where a user can specify an axis variable to do matplotlib methods.

    **plot_kwargs:** keyword arguments to add to the matplotlib plot() method.
    Many will not be accepted, but most stylistic arguments will be.
    """
    if plot_axis is None:
        plot_axis = plt
    # These are the arguments that are going to be put into plot_axis.plot that can be changed.
    final_plot_args = {
        "lw": None,
        "alpha": None,
        "figure": None,
        "ls": None,
        "marker": None,
        "mec": None,
        "mew": None,
        "mfc": None,
        "ms": None,
        "markevery": None,
        "solid_capstyle": None,
        "solid_joinstyle": None,
    }
    # This goes and modifies arguments from the default values of "None" to their respective values in plot_args.
    for key in plot_kwargs:
        if key in final_plot_args:
            final_plot_args[key] = plot_kwargs[key]
    # This is for the legend later on.
    graph_handles = []
    # This graphs all of the data in comparison_results.
    for key in comparison_results[0].keys():
        graph_handles.extend(
            plot_axis.plot(comparison_results[1], comparison_results[0][key], label=key, **final_plot_args)
        )
    # Now, we finally just create a legend.
    plt.legend(handles=graph_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def get_percent_correct(expected_values, predicted_values):
    """
    Takes in expected and predicted value arrays generated by the predict_data_with_known_type_with_classifier function
    and returns the percentage of predictions that are correct.

    **expected_values:** The array containing the expected values for a classifier to predict

    **predicted_values:** The array containing the values a classifier actually predicted

    **returns:** A float, from 0.0 to 100.0, that is the percentage of predictions that are correct.
    """
    # Gets the amount of values that are correct as a float.
    # It is a float so that integer division doesn't happen in the return statement.
    # A value is considered "correct" if the expected value is the same as the predicted value.
    num_correct = 0.0
    for val_index in range(len(predicted_values)):
        if expected_values[val_index] == predicted_values[val_index]:
            num_correct += 1.0
    # Now it just calculates the percent correct and returns it.
    return (num_correct / len(predicted_values)) * 100.0