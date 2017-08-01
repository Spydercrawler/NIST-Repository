from TableDictionarySystem import *
import pandas as pd

print "Reading table from csv!"
table = pd.read_csv("../Combined_Two_Port_Check_Standard.csv")
print "Splitting table by device_id column!"
dictionary_system = split_table_by_column(table, "Device_Id")
print "Splitting table by other columns!"
dictionary_system.split_by_columns("System_Id", "Measurement_Date")
print dictionary_system["CTN210"]["System 2,7"].keys()
print "Removing Duplicate Frequencies..."
dictionary_system.remove_column_duplicates("Frequency")
print dictionary_system['C00001']['HP8510']['9 Jul 2010']
print "Removing Short Tables!"
dictionary_system.remove_short_tables(10)
try:
    print dictionary_system["C10201"]["HIJ-WR22/1"].keys()
except KeyError:
    print "dictionary_system no longer has C10201"
print "Removing all unnecessary columns!"
dictionary_system.keep_only_certain_columns("Frequency", "magS11", "magS21", "magS22")
dictionary_system['C00001']['HP8510']['9 Jul 2010']
