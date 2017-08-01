from MachineLearningModule import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

table = pd.read_csv("Combined_Two_Port_Check_Standard.csv")
table_dictionary_system = split_table_by_columns(table, "Device_Id", "System_Id", "Measurement_Date")
table_dictionary_system.remove_column_duplicates("Frequency")
table_dictionary_system.remove_short_tables(10)
table_dictionary_system.keep_only_certain_columns("Frequency", "magS11", "magS21", "magS22")
value_system = table_dictionary_system.convert_to_array_dictionary_system()
value_system.interpolate_data(50, "Frequency", ("magS11", "magS21", "magS22"))
good_fake_data_system = value_system.make_fake_data_system_slope("Frequency",
                                                                 ("magS11", "magS21", "magS22"),
                                                                 15000,
                                                                 ("CTN210", "System 2,7"),
                                                                 [0.003, 0.0001, 0.001],
                                                                 [0.00220, 0.00015, 0.00441],
                                                                 [0.1, 0.08, 0.1])
bad_fake_data_system = value_system.make_fake_data_system_noise("Frequency",
                                                                ("magS11", "magS21", "magS22"),
                                                                15000,
                                                                ("CTN210", "System 2,7"),
                                                                (0.005, 0.001, 0.005))
frequency_vals = good_fake_data_system['Fake Dataset 1']["Frequency"]
good_magS11_arr = good_fake_data_system.get_dataset_variable_values("magS11")
good_magS21_arr = good_fake_data_system.get_dataset_variable_values("magS21")
good_magS22_arr = good_fake_data_system.get_dataset_variable_values("magS22")
bad_magS11_arr = bad_fake_data_system.get_dataset_variable_values("magS11")
bad_magS21_arr = bad_fake_data_system.get_dataset_variable_values("magS21")
bad_magS22_arr = bad_fake_data_system.get_dataset_variable_values("magS22")
magS11_classifier = train_classifier(classifier_type=RandomForestClassifier, good=good_magS11_arr, bad=bad_magS11_arr)
magS21_classifier = train_classifier(classifier_type=RandomForestClassifier, good=good_magS21_arr, bad=bad_magS21_arr)
magS22_classifier = train_classifier(classifier_type=RandomForestClassifier, good=good_magS22_arr, bad=bad_magS22_arr)
good_test_data_system = value_system.make_fake_data_system_slope("Frequency",
                                                                 ("magS11", "magS21", "magS22"),
                                                                 15000,
                                                                 ("CTN210", "System 2,7"),
                                                                 [0.003, 0.0001, 0.001],
                                                                 [0.00220, 0.00015, 0.00441],
                                                                 [0.1, 0.08, 0.1])
bad_test_data_system = value_system.make_fake_data_system_noise("Frequency",
                                                                ("magS11", "magS21", "magS22"),
                                                                15000,
                                                                ("CTN210", "System 2,7"),
                                                                (0.005, 0.001, 0.005))
test_magS11_arr = np.append(good_test_data_system.get_dataset_variable_values("magS11"),
                            bad_test_data_system.get_dataset_variable_values("magS11"), axis=0)
test_magS21_arr = np.append(good_test_data_system.get_dataset_variable_values("magS21"),
                            bad_test_data_system.get_dataset_variable_values("magS21"), axis=0)
test_magS22_arr = np.append(good_test_data_system.get_dataset_variable_values("magS22"),
                            bad_test_data_system.get_dataset_variable_values("magS22"), axis=0)
make_prediction_graph(magS11_classifier,
                      frequency_vals,
                      test_magS11_arr,
                      num_lines=1000,
                      z_indexes={"good": 1, "bad": 0})
plt.title("magS11 good vs bad predictions")
plt.show()
