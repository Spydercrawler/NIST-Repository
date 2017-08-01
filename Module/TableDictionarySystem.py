import pandas as pd
import numpy as np
from collections import Iterable
import copy
from ArrayDictionarySystem import *


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

    def items(self):
        """This returns the key,value pairs of the TableDictionarySystem's dictionary in tuples."""
        return self.dictionary.items()

    def values(self):
        """This returns a list of all the values in the TableDictionarySystem's dictionary."""
        return self.dictionary.values()

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
