# Some Basic Stuff:
import numpy as np
import pandas as pd
from collections import Iterable
import copy
from scipy import interpolate
import random
# For Smoothing:
from statsmodels.nonparametric.smoothers_lowess import lowess
# Sklearn Tools:
from sklearn.utils import shuffle
# Matplotlib stuff:
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# By the way, there are some other imports in the train_classifier method because they were going to be used
# in a specific circumstance so I decided to save a bit of space at the cost of a bit of time.


# Classes:
class TableDictionarySystem(object):
    # Functions:
    def __init__(self, basedict):
        # Raises an error if basedict is not a dictionary.
        if not isinstance(basedict, dict):
            raise ValueError("basedict must be a dictionary!")
        elif isinstance(basedict, dict):
            # Checks the validity of basedict. If basedict does not have the correct format,
            # or has value types that are not TableDictionarySystems, dictionaries,
            # or pandas Dataframes, raises a ValueError.
            if self.__check_dictionary_validity(basedict):
                self.dictionary = basedict
                # Converts all sub-dictionaries into TableDictionarySystems to
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
        # This is made to only accept TableDictionarySystems, dictionaries, and pandas DataFrames,
        # or else it will throw an error. I am sure you can figure this method out.
        if isinstance(value, (TableDictionarySystem, pd.DataFrame)):
            self.dictionary[key] = value
        elif isinstance(value, dict):
            self.dictionary[key] = TableDictionarySystem(value)
        else:
            raise ValueError("Value set is not a dictionary or pandas dataFrame!")

    def __len__(self):
        return len(self.dictionary)

    def __str__(self):
        return "TableDictionarySystem("+str(self.dictionary)+")"

    def __deepcopy__(self, memodict={}):
        return TableDictionarySystem(copy.deepcopy(self.dictionary))

    def __delitem__(self, key):
        del self.dictionary[key]

    # Old Dictionary Methods:
    def keys(self):
        """This returns the keys of the TableDictionarySystem's dictionary."""
        return self.dictionary.keys()

    def get(self, key, default=None):
        """
        Returns a value for a given key inside the ArrayDictionarySystem's dictionary.
        :param key: the key to be searched in the dictionary.
        :param default: The value to be returned in case the key specified does not exist.
        :return: the value for the given key in side the ArrayDictionarySystem's dictionary.
        """
        return self.dictionary.get(key, default)

    def pop(self, key, default=None):
        """
        Pops the value at the given key from the TableDictionarySystem's dictionary and returns it.
        :param key: The key to pop the value from
        :param default: If the given key does not exist within the TableDictionarySystem's dictionary,
        this value will be returned.
        :return: The value that was just popped from the given key.
        """
        return self.dictionary.pop(self, key, default=default)

    def items(self):
        """This returns the key,value pairs of the TableDictionarySystem's dictionary in tuples."""
        return self.dictionary.items()

    def values(self):
        """This returns a list of all the values in the TableDictionarySystem's dictionary."""
        return self.dictionary.values()

    def iteritems(self):
        """Returns the iterator returned by the iteritems method of the TableDictionarySystem's dictionary."""
        return self.dictionary.iteritems()

    def iterkeys(self):
        """Returns the iterator returned by the iterkeys method of the TableDictionarySystem's dictionary."""
        return self.dictionary.iterkeys()

    def itervalues(self):
        """Returns the iterator returned by the itervalues method of the TableDictionarySystem's dictionary."""
        return self.dictionary.itervalues()

    # New Methods:
    def split_by_column(self, column):
        """
        Splits the dictionary system's tables by a single column,
        making the location where the table was a dictionary containg unique column values in the column as keys,
        with each key containing a table.
        :param column: The column to split the dictionary systems by.
        """
        for key in self.keys():
            # If the value at that key is a TableDictionarySystem it calls split_by_column on that.
            if isinstance(self[key], TableDictionarySystem):
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
                self[key] = TableDictionarySystem(new_dict)
            # The method will raise an error if the TableDictionarySystem contains
            # values other than TableDictionarySystems and pandas DataFrames.
            else:
                raise ValueError("TableDictionarySystem contains values other than TableDictionarySystems and tables!")

    def split_by_columns(self, *columns):
        """
        Does the same thing as the split_by_column method, except splits by multiple columns in order instead of one.
        :param columns: The names of each column to split the tables by.
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
        except one in every table in the tabledictionarysystem,
        so that no duplicates in the said column will remain in the TableDictionarySystem.
        :param column: The column used to remove duplicate values.
        """
        for key in self.keys():
            # If the value at the current key is a TableDictionarySystem, it calls remove_column_duplicates on that.
            if isinstance(self[key], TableDictionarySystem):
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
            # The method will raise an error if the TableDictionarySystem contains
            # values other than TableDictionarySystems and pandas DataFrames.
            else:
                raise ValueError("DictionarySystem has a value that isn't a TableDictionarySystem or Pandas DataFrame!")

    def remove_short_tables(self, row_count):
        """
        Removes all tables in the TableDictionarySystem that are below a certain row count.
        :param row_count: The row count required for a table to stay in the TableDictionarySystem.
        """
        for key in self.keys():
            # If the value at the current key is a TableDictionarySystem, it calls remove_short_tables on that.
            # If the TableDictionarySystem has no tables left in it after that, the method deletes the current key.
            if isinstance(self[key], TableDictionarySystem):
                self[key].remove_short_tables(row_count)
                if self[key] is None or len(self[key]) <= 0:
                    del self[key]
            # If the value at the current key is a pandas DataFrame and it's row count is
            # less than the specified minimum row count, the method deletes the current key.
            elif isinstance(self[key], pd.DataFrame):
                if self[key].shape[0] < row_count:
                    del self[key]
            # If the value at the current key is not a TableDictionarySystem or pandas DataFrame,
            # the method raises a ValueError.
            else:
                raise ValueError("DictionarySystem has a value that isn't a TableDictionarySystem or Pandas DataFrame!")

    def keep_only_certain_columns(self, *columns):
        """
        Removes all columns in every table except the ones specified.
        This is generally used to remove irrelevant data from tables.
        :param columns: The names of the columns to keep in every table.
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
        # going through the keys in the TableDictionarySystem.
        for key in self.keys():
            # If the value at the current key is a TableDictionarySystem,
            # the keep_only_certain_columns method is called on that.
            if isinstance(self[key], TableDictionarySystem):
                self[key].keep_only_certain_columns(columns_flattened)
            # If the value a the current key is a DataFrame, the method sets the value at the current key to a new
            # DataFrame that only has the columns specified.
            elif isinstance(self[key], pd.DataFrame):
                self[key] = self[key][columns_flattened]
            # If the value at the current key is not a TableDictionarySystem or pandas DataFrame,
            # the method will raise an error.
            else:
                raise ValueError(
                    "DictionarySystem has a value that isn't a TableDictionarySystem or Pandas DataFrame!")

    def convert_to_array_dictionary_system(self):
        """
        Returns an array dictionary system containing the values in every table.
        The array dictionary system is generally used for fake data generation and interpolation.
        :return: An array dictionary system containing the values in every table.
        """
        # First, we perform a deep copy of our dictionary, because we don't want our original TableDictionarySystem to
        # be turned into the format of the ArrayDictionarySystem.
        array_dictionary_system = copy.deepcopy(self.dictionary)
        # Then we do all of the action with the following method, because it involves recursion and
        # it may be difficult to implement that with this method
        self.__convert_sub_dictionary_to_array_system(array_dictionary_system)
        # Then it is converted into an ArrayDictionarySystem and returned.
        return ArrayDictionarySystem(array_dictionary_system)

    def multiple_edits(self, **kwargs):
        """
        Does multiple TableDictionarySystem methods in a single method;
        This is used in case a user wants concise or small code.
        :param kwargs: The methods and arguments to call.
        The keywords must be the names of other TableDictionarySystem methods,
        and their values must be dictionaries containing the arguments.
        """
        # Honestly, you can probably figure this method out, so I am just not going to comment this one.
        if "split_by_column" in kwargs:
            self.split_by_column(kwargs["split_by_column"]["column"])
        if "split_by_columns" in kwargs:
            self.split_by_columns(kwargs["split_by_columns"]["columns"])
        if "remove_column_duplicates" in kwargs:
            self.remove_column_duplicates(kwargs["remove_column_duplicates"]["column"])
        if "remove_short_tables" in kwargs:
            self.remove_short_tables(kwargs["remove_short_tables"]["row_count"])
        if "keep_only_certain_columns" in kwargs:
            self.keep_only_certain_columns(*kwargs["keep_only_certain_columns"]["columns"])

    # Helper Methods:
    def __convert_sub_dictionary_to_array_system(self, dictionary):
        """
        Private method which converts any tables in a nested dictionary system into
        dictionaries containing arrays of values at each key, with keys corresponding to columns.
        :param dictionary: The dictionary to go through to convert tables in.
        """
        for key in dictionary.keys():
            # If the value at the current key is a TableDictionarySystem it gets converted into a dictionary.
            if isinstance(dictionary[key], TableDictionarySystem):
                dictionary[key] = dictionary[key].dictionary
            # If the value at the current key is a dictionary,
            # it has the __convert_sub_dictionary_to_array_system method called on it.
            if isinstance(dictionary[key], dict):
                self.__convert_sub_dictionary_to_array_system(dictionary[key])
            # If the value at the current key is a pandas DataFrame, it starts converting it to an array thing.
            elif isinstance(dictionary[key], pd.DataFrame):
                # First, it calls the DataFrame to_dict method to convert the DataFrame into a dictionary of lists.
                dictionary[key] = dictionary[key].to_dict(orient="list")
                # Then, it converts each list into a numpy array.
                for subkey in dictionary[key].keys():
                    if isinstance(dictionary[key][subkey], list):
                        dictionary[key][subkey] = np.array(dictionary[key][subkey])
        # Because dictionaries are mutable, this does not have to return anything to have its desired effect.

    def __check_dictionary_validity(self, dictionary):
        """
        Checks if a dictionary has a valid format and valid variable types to become a TableDictionarySystem.
        :param dictionary: The dictionary to check.
        :return: True or False depending on whether a dictionary has a valid format and valid variable types.
        """
        # If the dictionary variable inputted is not a dictionary, returns False.
        if isinstance(dictionary, (TableDictionarySystem, dict)):
            for key in dictionary.keys():
                # If any value in the dictionary is not a TableDictionarySystem, dictionary, or
                # pandas DataFrame, returns False.
                if isinstance(dictionary[key], (TableDictionarySystem, dict)):
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
        Converts all sub-dictionaries in the TableDictionarySystem's main dictionary into TableDictionarySystems.
        """
        # Literally all this does is if any value in the TableDictionarySystem is a dictionary, it gets turned into a
        # TableDictionarySystem.
        for key in self.keys():
            if isinstance(self[key], dict):
                self[key] = TableDictionarySystem(self[key])

    def __better_flatten(self, iterable_object):
        """Some flattening method I found on stackoverflow, which I a few methods
        to make the flattening of my columns arguments work."""
        for value in iterable_object:
            if isinstance(value, Iterable) and not isinstance(value, basestring):
                for newvalue in self.__better_flatten(value):
                    yield newvalue
            else:
                yield value


class ArrayDictionarySystem(object):

    # Functions:
    def __init__(self, basedict):
        # Raises an error if basedict is not a dictionary.
        if not isinstance(basedict, dict):
            raise ValueError("basedict must be a dictionary!")
        elif isinstance(basedict, dict):
            # Checks the validity of basedict. If basedict does not have the correct format,
            # or has value types that are not ArrayDictionarySystems, dictionaries,
            # or numpy arrays, raises a ValueError.
            if self.__check_dictionary_validity(basedict):
                self.dictionary = basedict
                # Converts all sub-dictionaries into ArrayDictionarySystems to
                # allow all of the recursive methods to work.
                self.__convert_dicts_to_systems()
            else:
                raise ValueError("Basedict is not in a valid format, or has invalid values!")
        # This is just in case something goes dreadfully wrong, but hopefully it will never be called.
        else:
            raise ValueError("Alright, I don't know what you even did, but something is wrong with basedict.")

    def __getitem__(self, key):
        return self.dictionary[key]

    def __setitem__(self, key, value):
        # This is made to only accept ArrayDictionarySystems, dictionaries, and numpy arrays,
        # or else it will throw an error. I am sure you can figure this method out.
        if isinstance(value, (ArrayDictionarySystem, np.ndarray)):
            self.dictionary[key] = value
        elif isinstance(value, dict):
            self.dictionary[key] = ArrayDictionarySystem(value)
        else:
            raise ValueError("Value set is not a dictionary or numpy array!")

    def __len__(self):
        return len(self.dictionary)

    def __str__(self):
        return str(self.dictionary)

    def __deepcopy__(self, memodict={}):
        return ArrayDictionarySystem(copy.deepcopy(self.dictionary))

    def __delitem__(self, key):
        del self.dictionary[key]

    # Old Dictionary Methods
    def keys(self):
        """This returns the keys of the TableDictionarySystem's dictionary."""
        return self.dictionary.keys()

    def values(self):
        """This returns a list of all the values in the TableDictionarySystem's dictionary."""
        return self.dictionary.values()

    def items(self):
        """This returns the key,value pairs of the ArrayDictionarySystem's dictionary in tuples."""
        return self.dictionary.items()

    def get(self, key, default=None):
        """
        Returns a value for a given key inside the ArrayDictionarySystem's dictionary.
        :param key: the key to be searched in the dictionary.
        :param default: The value to be returned in case the key specified does not exist.
        :return: the value for the given key in side the ArrayDictionarySystem's dictionary.
        """
        return self.dictionary.get(key, default)

    def pop(self, key, default=None):
        """
        Pops the value at the given key from the ArrayDictionarySystem's dictionary and returns it.
        :param key: The key to pop the value from
        :param default: If the given key does not exist within the ArrayDictionarySystem's dictionary,
        this value will be returned.
        :return: The value that was just popped from the given key.
        """
        return self.dictionary.pop(self, key, default=default)

    def iteritems(self):
        """Returns the iterator returned by the iteritems method of the ArrayDictionarySystem's dictionary."""
        return self.dictionary.iteritems()

    def iterkeys(self):
        """Returns the iterator returned by the iterkeys method of the ArrayDictionarySystem's dictionary."""
        return self.dictionary.iterkeys()

    def itervalues(self):
        """Returns the iterator returned by the itervalues method of the ArrayDictionarySystem's dictionary."""
        return self.dictionary.itervalues()

    # New Methods:
    def interpolate_data(self,
                         num_points,
                         independent_variable,
                         dependent_variables,
                         progress_printing=False,
                         interpolation_kind="cubic"):
        """
        Goes through the ArrayDictionarySystem and interpolates all the datasets to be of the length specified.
        This will remove all arrays in the ArrayDictionarySystem that are not independent or dependent variables,
        because I couldn't figure out a reliable way to not remove them. Also, keep in mind that there must be an
        independent variable for this to work.
        :param num_points: The number of points to be interpolated to.
        :param independent_variable: The name of the independent variable to be used during interpolation.
        The values of this variable must be some type of number.
        :param dependent_variables: An iterable containing the names of the dependent variables to be interpolated.
        :param progress_printing: Whether or not to print out the progress on interpolating the function;
        This is mostly used for debugging.
        :param interpolation_kind: The kind of interpolation to be used.
        Values that will not throw an error are all values accepted by scipy's interp1d function.
        """
        for key in self.keys():
            # This is a bit messy, but this if statement checks to see if the current location is an
            # ArrayDictionarySystem containing individual dataset ArrayDictionarySystems.
            if isinstance(self[key].values()[0].values()[0], np.ndarray):
                # So the interpolation range for all of the datasets in the ArrayDictionarySystem,
                # to include as many datasets as possible,
                # has to have the minimum value be the maximum value of all minimum values of the existing datasets,
                # and has to have the maximum value be the minimum value
                #  of all maximimum values of the existing datasets. Here we start finding these values.
                maximum_of_mins = None
                minimum_of_maxes = None
                for dataset_key in self[key].keys():
                    if independent_variable not in self[key][dataset_key].keys():
                        raise KeyError("Independent variable specified is not a key in the ArrayDictionarySystem!")
                    # All of this here is simple code to find these maxes of minimums and minimums of maxes:
                    if maximum_of_mins is None:
                        maximum_of_mins = self[key][dataset_key][independent_variable].min()
                    elif self[key][dataset_key][independent_variable].min() > maximum_of_mins:
                        maximum_of_mins = self[key][dataset_key][independent_variable].min()
                    if minimum_of_maxes is None:
                        minimum_of_maxes = self[key][dataset_key][independent_variable].max()
                    elif self[key][dataset_key][independent_variable].max() < minimum_of_maxes:
                        minimum_of_maxes = self[key][dataset_key][independent_variable].max()
                # Now that we have the bounds for our interpolation, we need to make our interpolation functions:
                for dataset_key in self[key].keys():
                    # First we need to get the values for the independent and
                    # dependent variables in this specific dataset:
                    independent_var_values = self[key][dataset_key][independent_variable].tolist()
                    dependent_var_values = [
                        self[key][dataset_key][variable_name].tolist() for variable_name in dependent_variables]
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
                        # Finally, we replace our old ArrayDictionarySystem for this dataset with
                        # our newly interpolated one. By the way, we are inputting a dictionary, but the __setitem__
                        # method will convert it into an ArrayDictionarySystem, so we do not have to worry about that.
                        self[key][dataset_key] = interpolated_dictionary
                        if progress_printing:
                            print "Completed Dictionary Key!"
                    # Sometimes interpolation fails because some dataset's bounds are outside of the
                    # interpolation range we just got, so this is here to just remove all the datasets that do that.
                    except ValueError:
                        if progress_printing:
                            print "A Dictionary Key had an error, so it was removed!"
                        del self[key][dataset_key]
                        continue
            # If the current ArrayDictionarySystem is not a ArrayDictionarySystem
            # containing ArrayDictionarySystems for individual datasets, the interpolate_data
            # method is called for it; as a result, this recursion will eventually interpolate all the data
            elif isinstance(self[key].values()[0].values()[0], ArrayDictionarySystem):
                self[key].interpolate_data(num_points, independent_variable, dependent_variables)
            else:
                raise ValueError("Something is in the ArrayDictionarySystem that is not "
                                 "a numpy array or another ArrayDictionarySystem!")

    def make_fake_data_system_noise(self,
                                    independent_var,
                                    dependent_vars,
                                    num_datasets,
                                    location,
                                    randomness_amplitudes,
                                    progress_printing=False):
        """
        Returns an ArrayDictionarySystem containing fake data created by
        adding random noise to data in the original ArrayDictionarySystem.
        :param independent_var: The name independent variable of the variables specified.
        The independent variable's values will be kept constant, and noise will not be added to it.
        :param dependent_vars: A list containing the names of
        the dependent variables of the variables in the ArrayDictionarySystem.
        Noise will be added to these to create fake data.
        :param num_datasets: The amount of datasets to be created at each dictionary of datasets
        at the end of the nesting in the nested dictionary in the ArrayDictionarySystem.
        Honestly, I really don't know how to explain that whole "end of nesting dictionary" thing,
        it is much easier to show
        :param location: The location to start the fake data creation.
        This is used in case, a user wants to make a fake data system for only one index of the ArrayDictionarySystem.
        If the user wishes create fake data for the entire ArrayDictionarySystem, they can input None for location.
        :param randomness_amplitudes: A list containing the amplitudes of the random noise
        for each of the dependent variables, in order.
        :param progress_printing: A boolean specifying whether or not to print the progress of the fake data creation.
        This is generally made true for debugging purposes.
        :return: A new ArrayDictionarySystem containing fake data generated by the method.
        """
        # Location can either be an iterable or a string, so this checks to see if location is a valid iterable:
        if isinstance(location, Iterable) and len(location) > 0 and not isinstance(location, basestring):
            # If it's length is one, that means that means that all of the values in the current ArrayDictionarySystem
            # are SUPPOSED to be ArrayDictionarySystems representing singular datasets.
            if len(location) == 1:
                # the current_location variable is just there so I don't have to keep writing location[0] a ton.
                current_location = location[0]
                # new_data_dict will contain all of the new, fake datasets.
                new_data_dict = {}
                for i in range(num_datasets):
                    # We need a deep copy of one of the datasets so that
                    # when we add random noise it won't affect the original dataset:
                    new_set_values = copy.deepcopy(self[current_location][random.choice(self[current_location].keys())])
                    # Now we add random noise to every single dependent variable:
                    for dependent_var, amp in zip(dependent_vars, randomness_amplitudes):
                        num_values_in_dataset = len(new_set_values[dependent_var])
                        new_set_values[dependent_var] += np.random.normal(0, amp, num_values_in_dataset)
                    # We now have a fake dataset, so we add it to new_data_dict, which contains all of our fake datasets
                    new_data_dict["Fake Dataset " + str(i)] = new_set_values
                    # If we have progress printing on, we can print our progress if the time is right:
                    if progress_printing and i % (num_datasets // 10) == 0:
                        print str(i) + " fake datasets have been created!"
                # ...and now we just return new_data_dict as an ArrayDictionarySystem!
                return ArrayDictionarySystem(new_data_dict)
            else:
                # if the length of location is more than one, we simply use recursion to narrow down the location:
                return self[location[0]].make_fake_data_system_noise(independent_var,
                                                                     dependent_vars,
                                                                     num_datasets,
                                                                     location[1:],
                                                                     randomness_amplitudes,
                                                                     progress_printing)
        # If location is a string, the same method is called, but location is now a single element list,
        # so that The above code can do it's magic.
        elif isinstance(location, basestring):
            return self.make_fake_data_system_noise(independent_var,
                                                    dependent_vars,
                                                    num_datasets,
                                                    [location],
                                                    randomness_amplitudes,
                                                    progress_printing=progress_printing)
        # If location is not an iterable or a string, a ValueError is raised.
        else:
            raise ValueError("Location must be an iterable or a string!")

    def make_fake_data_system_slope(self,
                                    independent_var,
                                    dependent_vars,
                                    num_datasets,
                                    location,
                                    starting_noises,
                                    slope_deviations,
                                    smoothing_fracs,
                                    progress_printing=False):
        """
        Produces Fake Data by adding a single point of random noise to the first point of an existing dataset, then adds
        random deviations to the slope of every line segment in said existing dataset to create a new, fake dataset.
        :param independent_var: The name of the independent variable.
        The values of the independent variable will not be changed.
        :param dependent_vars: An Iterable containing the names of the dependent variables. The values of the dependent
        variables will be modified from the values of existing datasets to create fake datasets.
        :param num_datasets: The amount of fake datasets to create.
        :param location: The location to start the fake data creation.
        This is used to make a fake data system for only one end index of the ArrayDictionarySystem, because making fake
        data for an entire ArrayDictionarySystem can often crash a python program due to the massive variable size.
        :param starting_noises: An Iterable containing the noises to add to every starting value in existing datasets to
        start the creation of fake datasets. Each value of the iterable corresponds to the noise to add to a certain
        dependent variable's data.
        :param slope_deviations: An Iterable containing the noises to add to the slopes of existing datasets to create
        fake datasets. Each value of the iterable corresponds to the noise to add to the slopes of a
        certain dependent variable's data.
        :param smoothing_fracs: How much to smooth the lines after they are created. Don't overdo this;
        a smoothing frac of 0.1 is quite a lot.
        :param progress_printing: A boolean specifying whether or not to print the progress of the fake data creation.
        This is generally made true for debugging purposes.
        :return: A new ArrayDictionarySystem containing fake data generated by the method.
        """
        # I can comment more of this later, right now I am tired, so I will comment the bare minimum.
        # If someone other than me is reading this, then I screwed up, I am sorry.
        if isinstance(location, Iterable) and len(location) > 0 and not isinstance(location, basestring):
            if len(location) == 1:
                current_location = location[0]
                new_data_dict = {}
                for fake_set_num in range(num_datasets):
                    # Copying a random dataset in our already existing datasets,
                    # which will be modified to become fake data.
                    new_set_values = copy.deepcopy(self[current_location][random.choice(self[current_location].keys())])
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
                    if progress_printing and fake_set_num%(num_datasets//10) == 0:
                        print str(fake_set_num) + " fake datasets have been created!"
                return ArrayDictionarySystem(new_data_dict)
            else:
                # if the length of location is more than one, we simply use recursion to narrow down the location:
                return self[location[0]].make_fake_data_system_slope(independent_var,
                                                                     dependent_vars,
                                                                     num_datasets,
                                                                     location[1:],
                                                                     starting_noises,
                                                                     slope_deviations,
                                                                     smoothing_fracs,
                                                                     progress_printing=progress_printing)
        elif isinstance(location, basestring):
            return self[location[0]].make_fake_data_system_slope(independent_var,
                                                                 dependent_vars,
                                                                 num_datasets,
                                                                 [location],
                                                                 starting_noises,
                                                                 slope_deviations,
                                                                 smoothing_fracs,
                                                                 progress_printing=progress_printing)
        else:
            raise ValueError("Location must be an iterable or a string!")

    def get_dataset_variable_values(self, dataset_variable, location=None):
        """
        This method is intended to get data in a format to be put into a classifier,
        and returns an array with a lot of lists containing datapoints of the dataset variable specified.
        Each list corresponds to the datapoints for a single dataset.
        :param dataset_variable: The variable that will have its data returned in an array.
        The default way this data is stored is in an array containing a lot of lists.
        Each list stores the datapoints of the variable in a single dataset.
        :param location: The location within the ArrayDictionarySystem you want to get the data from.
        The default is none, which gets data from the entire array,
        however you can also put in a string as a key or an iterable of string keys which will be accessed in a row,
        in case you need to get something nested far within the ArrayDictionarySystem.
        :return: An array containing a lot of lists, each list containing the datapoints of the variable from a single dataset.
        """
        # If no location was inputted, it simply gets all of the dataset
        # variable values in every dataset and returns an array of those.
        if location is None:
            # This list will contain the list of values of the dataset variable for every dataset in
            # the ArrayDictionarySystem.
            variable_set_list = []
            for key in self.keys():
                # If the value at the first key in the current dictionary is an array,
                # it adds the array whose key is the dataset variable name to variable_set_list
                if isinstance(self[key].values()[0], np.ndarray):
                    variable_set_list.append(self[key][dataset_variable].tolist())
                # if the current location is an ArrayDictionarySystem that contains ArrayDictionarySystems representing
                # individual datasets, it goes through them calls get_dataset_variable_values to each one,
                # then adds them all to variable_set_list.
                elif isinstance(self[key].values()[0].values()[0], np.ndarray):
                    key_results = self[key].get_dataset_variable_values(dataset_variable, location=()).tolist()
                    if isinstance(key_results[0], list):
                        variable_set_list.extend(key_results)
                    else:
                        variable_set_list.append(key_results)
                # if the current location is higher up in the nesting, the method calls get_dataset_variable_values on
                # the ArrayDictionarySystem at the current location and adds the results to variable_set_list.
                else:
                    key_results = self[key].get_dataset_variable_values(dataset_variable, location=None).tolist()
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
            if len(basestring) == 0:
                return self.get_dataset_variable_values(dataset_variable, location=tuple())
            return self.get_dataset_variable_values(dataset_variable, location=tuple(basestring))
        # If location is an iterable, but not a string, then that means we are looking at some sort of path.
        elif isinstance(location, Iterable) and not isinstance(location, basestring):
            # If the length of location is 0, then that means that the current location is the location to
            # get the dataset variable's values from, so we go through the ArrayDictionarySystem at the current location
            # and add all of the values of the dataset variable to a list, then return it as an array.
            if len(location) == 0:
                variable_set_list = []
                for key in self.keys():
                    variable_set_list.append(self[key][dataset_variable].tolist())
                return np.array(variable_set_list)
            # If the length of location is 1 then we can do the same thing as when location is 0 with a bit of tweaking.
            elif len(location) == 1:
                current_location = location[0]
                variable_set_list = []
                for key in self[current_location].keys():
                    variable_set_list.append(self[current_location][key][dataset_variable].tolist())
                return np.array(variable_set_list)
            # If the length of location is greater than one then we just use recursion to our advantage.
            else:
                return self[location[0]].get_dataset_variable_values(dataset_variable, location[1:])
        # If location is not None, a string, or an Iterable, we raise a ValueError.
        else:
            raise ValueError("location must be an Iterable!")

    def remove_variable(self, variable):
        """
        Takes in a variable name as an argument, then removes all instances of that
        variable from any datasets in the ArrayDictionarySystem.
        :param variable: The name of the variable to remove from all datasets in the ArrayDictionarySystem.
        """
        # If the current location is an ArrayDictionarySystem containing individual datasets,
        # it goes through them and deletes the variable from each one.
        if isinstance(self.values()[0].values()[0], np.ndarray):
            for key in self.keys():
                del self[key][variable]
            return
        # Otherwise, if it is not in such a location, it simply calls remove_variable
        # on each of the sub-ArrayDictionarySystem.
        for key in self.keys():
            self[key].remove_variable(variable)

    def remove_variables(self, *variables):
        """
        Takes in multiple variable names as arguments and removes
        each one of these variables from every dataset in the ArrayDictionarySystem.
        :param variables: the names of every variable to be removed.
        """
        # All this does is call remove_variable for each of the variables specified.
        for variablename in variables:
            self.remove_variable(variablename)

    def keep_only_certain_variables(self, *variables):
        """
        Removes all variables in the ArrayDictionarySystem except the ones specified.
        :param variables: The names of the variables to be kept in the ArrayDictionarySystem.
        """
        # Since we don't want to have an empty ArrayDictionarySystem,
        # we will raise an error if no variables are specified.
        if len(variables) == 0:
            raise ValueError("At least one variable name must be specified.")
        # This if statement checks to see if our current location is an
        # ArrayDictionarySystem containing individual datasets.
        if isinstance(self.values()[0].values()[0], np.ndarray):
            # For every dataset, go through every key in the dataset's dictionary and
            # add the values of the specified variables to a new dictionary, which replaces the old dictionary.
            for key in self.keys():
                new_dict = {}
                for var in variables:
                    if var in self[key].keys():
                        new_dict[var] = self[key][var]
                    else:
                        raise ValueError('One of the variables specified did not exist in a dataset!')
                self[key] = new_dict
            return
        # If we are not currently at an ArrayDictionarySystem containing individual datasets,
        # we simply call the same method on all of the sub-ArrayDictionarySystems.
        for key in self.keys():
            self[key].keep_only_certain_variables(*variables)

    # Helper Methods:
    def __check_dictionary_validity(self, dictionary):
        """
        Checks whether a dictionary inputted is a valid dictionary for the creation of an ArrayDictionarySystem.
        :param dictionary: The dictionary to be checked.
        :return: A boolean detailing whether or not the dictionary is
        a valid dictionary for the creation of an ArrayDictionarySystem.
        """
        # If the dictionary variable inputted is not a dictionary, returns False.
        if isinstance(dictionary, (ArrayDictionarySystem, dict)):
            for key in dictionary.keys():
                # If any value in the dictionary is not a TableDictionarySystem, dictionary, or
                # numpy Array, returns False.
                if isinstance(dictionary[key], (ArrayDictionarySystem, dict)):
                    # If the __check_dictionary_validity method fails on any sub dictionaries, returns False.
                    if not self.__check_dictionary_validity(dictionary[key]):
                        return False
                elif isinstance(dictionary[key], np.ndarray):
                    continue
                else:
                    return False
        else:
            return False
        # If it never returned False, returns True.
        return True

    def __convert_dicts_to_systems(self):
        """
        Converts sub-dictionaries within the ArrayDictionarySystem into ArrayDictionarySystems.
        This is used during the initialization of the ArrayDictionarySystem.
        """
        # This just converts any sub-dictionaries into ArrayDictionarySystems. That's it.
        for key in self.keys():
            if isinstance(self[key], dict):
                self[key] = ArrayDictionarySystem(self[key])


# Table Splitting Methods:
def split_table_by_column(table, column):
    """
    Takes a pandas DataFrame and splits it by column values into a TableDictionarySystem.
    :param table: The Dataframe to split into a TableDictionarySystem.
    :param column: The column to split the pandas DataFrame by.
    :return: A TableDictionarySystem created by splitting the pandas DataFrame inputted by the column inputted
    """
    table_dictionary = {}
    # Goes through every unique value in the specified column of the the table inputted
    unique_column_vals = table[column].unique()
    for val in unique_column_vals:
        # Adds to the dictionary a new table with every value in the
        # specified column being the current unique value from the for loop.
        table_dictionary[val] = table[table[column] == val]
    return TableDictionarySystem(table_dictionary)


def split_table_by_columns(table, *columns):
    """
    Takes a pandas DataFrame and splits it by all of the columns specified into a TableDictionarySystem
    :param table: The Dataframe to split into a TableDictionarySystem.
    :param columns: The columns to split the pandas DataFrame by.
    :return: A TableDictionarySystem created by splitting the pandas DataFrame inputted by the columns inputted
    """
    # Raises a ValueError if no columns are inputted:
    if len(columns) == 0:
        raise ValueError("At least one column must be specified!")
    # Splits the table by the first column:
    table_dictionary_system = split_table_by_column(table, columns[0])
    # If more than one column was specified, split the TableDictionarySystem by the extra columns as well.
    if len(columns) > 1:
        table_dictionary_system.split_by_columns(columns[1:])
    return table_dictionary_system


def split_csv_by_column(path, column):
    """
    Takes a path to a csv file, loads it as a pandas dataframe, splits it by a column,
    and returns the resulting TableDictionarySystem.
    :param path: The path to a csv file.
    :param column: The column to split the pandas dataframe by.
    :return: A TableDictionarySystem that is the result of splitting the
    table contained in the csv file specified by the columns specified.
    """
    pandas_dataframe = pd.read_csv(path)
    return split_table_by_column(pandas_dataframe, column)


def split_csv_by_columns(path, *columns):
    """
    Takes a path to a csv file, loads it as a pandas dataframe, splits it by multiple columns,
    and returns the resulting TableDictionarySystem.
    :param path: The path to a csv file.
    :param columns: The columns to split the pandas dataframe by.
    :return:
    """
    pandas_dataframe = pd.read_csv(path)
    return split_table_by_columns(pandas_dataframe, *columns)


# Actions for value Arrays:
def plot_variable_array(independent_variable_arr,
                        dependent_variable_arr,
                        num_lines="all",
                        plot_axis=plt,
                        **plot_kwargs):
    """
    This method takes two arrays containing lists of values and graphs lines from the information in the arrays.
    :param independent_variable_arr: a two dimensional array containing lists of values for the x-axis.
    :param dependent_variable_arr: a two dimensional array containing lists of values for the y-axis
    :param num_lines: The amount of lines to be graphed.
    :param plot_axis: Optional argument, where a user can specify an axis variable to do matplotlib methods.
    :param plot_kwargs: keyword arguments to add to the matplotlib plot() method.
    Many will not be accepted, but most stylistic arguments will be.
    """
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
    # Before the method actually graphs lines, it checks to see if the amount of lines the user wants to graph
    # is bigger than the amount of lines that are possible to be graphed, just in case.
    if new_num_lines <= len(independent_variable_arr) and new_num_lines <= len(dependent_variable_arr):
        # The method plots the amount of lines specified by the method call.
        for i in range(new_num_lines):
            plot_axis.plot(independent_variable_arr[i], dependent_variable_arr[i], **final_plot_args)
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
        # Well, this covers that! Using the check_estimator method, we can tell if the inputted classifier is valid,
        # and raise an error if it isn't.
        # Keep in mind I am importing this here because if the user never inputs a classifier's class for
        # classifier_type, this import would never be used.
        from sklearn.utils.estimator_checks import check_estimator
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
                          plot_axis=plt,
                          z_indexes=None,
                          **plot_kwargs):
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
    :param plot_axis: Optional argument, where a user can specify an axis variable to do matplotlib methods.
    :param z_indexes: a dictionary with keys being all of the kwarg keys,
    which contains z indexes for different predicted results.
    If not specified or set to None, the default z indexes will be used.
    :param plot_kwargs: keyword arguments to add to the matplotlib plot() method.
    Many will not be accepted, but most stylistic arguments will be.
    """
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
    This is a small method which is used in some other methods which,
    if number_string is an integer within a string, returns that integer,
    and if number_string is an indication of percentage, like "tenth" or "seventh" it returns that amount of max_values.
    For instance, if "third" is inputted for number_string, and max_values is thirty, 10 would be returned.
    :param number_string: The string containing the integer or indication of percentage.
    :param max_values: The maximum amount of values in a dataset or something,
    this is used for strings that indicate percentage.
    :return: The integer the number string contains or the integer calculated by the percentage number_string contains.
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
    Since the predict_data_with_known_type_with_classifier method only returns two lists, this method exists to get
    a set of multiple data arrays into the same list format as the predict_data_with_known_type_with_classifier method.
    :param data_arrays: The arrays to turn into a list.
    :return: the combined list of all of the data arrays.
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
                                       classifiers,
                                       progress_printing=False):
    """
    Takes an array of 'good' datasets, as well as an array of random noise amplitudes to test and classifier types
    as keyword arguments, and returns data to be graphed in the graph_comparison_results.
    These two methods together will make a graph showing the percentage of guesses that were correct at
    different random noise amplitudes for different classifiers. Users can use this to compare the performance
    of different classifiers. The reason why these methods is because since this process takes a large amount of time,
    so if the methods are separated, it becomes a lot easier to quickly modify a plot to one's liking if the two methods
    are in an ipython notebook or something similar.
    :param good_datasets: An array of good datasets for a single variable.
    :param randomness_amplitude_range: An array all of the random noise amplitudes to use for "bad" data in tests.
    :param progress_printing: Whether or not to print the method's progress. This is mainly used for debugging.
    :param classifiers: A dictionary containing the classifier's CLASSES, not classifier objects, to test.
    Each key must be the name of each classifier.
    :return: Data to be used by the graph_comparison_results.
    """
    # This will contain the results at each point for each classifier.
    classifier_results = {}
    # This will contain all the amplitudes later.
    final_range = []
    if progress_printing:
        print "Graphing Progress: ",
    # Each key of classifier_results will be the name of a classifier, and each value will be a list of floats,
    # each one detailling the percentage of predictions the classifier got correct at a specific point.
    for key in classifiers.keys():
        classifier_results[key] = []
    for amplitude in randomness_amplitude_range:
        # Now we generate an array of bad datasets, from the good ones.
        # We have to do this manually, since this isn't an ArrayDictionarySystem and we don't have a method for it.
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
        if progress_printing:
                print "\b#",
    # Now we just return a list containing classifier_results and final_range.
    return [classifier_results, final_range]


def graph_comparison_results(comparison_results,
                             plot_axis=plt,
                             **plot_args):
    """
    This takes in the data that was generated by the get_classifier_comparison_results method, and plots it into a
    graph comparing classifier results. The reason why these methods is because the get_classifier_comparison_results
    method takes a lot of time, so if the methods are separated, it becomes a lot easier to quickly modify a plot to
    one's liking if the two methods are in an ipython notebook or something similar.
    :param comparison_results: The results of the get_classifier_comparison_results method.
    :param plot_axis: Optional argument, where a user can specify an axis variable to do matplotlib methods.
    :param plot_args: keyword arguments to add to the matplotlib plot() method.
    Many will not be accepted, but most stylistic arguments will be.
    """
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
    for key in plot_args:
        if key in final_plot_args:
            final_plot_args[key] = plot_args[key]
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
    Takes in expected and predicted value arrays generated by the predict_data_with_known_type_with_classifier method
    and returns the percentage of predictions that are correct.
    :param expected_values: The array containing the expected values for a classifier to predict
    :param predicted_values: The array containing the values a classifier actually predicted
    :return: A float, from 0.0 to 100.0, that is the percentage of predictions that are correct.
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
