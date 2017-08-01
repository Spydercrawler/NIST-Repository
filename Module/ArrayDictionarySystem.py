import numpy as np
from collections import Iterable
import copy
from scipy import interpolate
import random
from statsmodels.nonparametric.smoothers_lowess import lowess


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
                    dependent_var_values = [self[key][dataset_key][variable_name].tolist() for variable_name in dependent_variables]
                    # This list will hold the interpolation functions soon:
                    dependent_var_interpolation_functions = []
                    for dependent_var_list in dependent_var_values:
                        # Now, we simply use scipy's interp1d function to get our interpolation functions:
                        var_function = interpolate.interp1d(independent_var_values, dependent_var_list, interpolation_kind)
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
                raise ValueError("Something is in the ArrayDictionarySystem that is not a numpy array or another ArrayDictionarySystem!")

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
        This is used to make a fake data system for only one end index of the ArrayDictionarySystem, because making fake
        data for an entire ArrayDictionarySystem can often crash a python program due to the massive variable size.
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
                    # Now we just add random noise to every point in our dataset:
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
