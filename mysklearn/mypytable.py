from mysklearn import myutils

from mysklearn import myutils

import copy
import csv
from gettext import find
from multiprocessing.sharedctypes import Value
from pprint import PrettyPrinter
from re import M, T
from runpy import _TempModule
from sre_compile import isstring
from typing import NewType
from tabulate import tabulate
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
         """Prints the table in a nicely formatted grid structure.
         """
         print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = []
        col_index = 0
        copy_table = MyPyTable(self.column_names, self.data)

        if include_missing_values == True:
            pass
        else:
            copy_table.remove_rows_with_missing_values() # removes all missing vals

        #col_identifier is an Int
        if isinstance(col_identifier, int) == True:
            if col_identifier >= len(copy_table.column_names):
                raise IndexError("Column Identifer '{}' is either too large or too small, please try again".format(col_identifier))
            else:
                for row in copy_table.data:
                    col.append(row[col_identifier]) # append value in the row at the column we want into the column list
        #col_identifier is a String
        elif isstring(col_identifier) == True:
            for i in range(len(copy_table.column_names)):
                if copy_table.column_names[i] == col_identifier:
                    col_index = i
                    break
                else:
                    if i == len(copy_table.column_names) - 1:
                        raise ValueError("Column: '{}' could not be found, please try again".format(col_identifier))

            for row in copy_table.data:
                col.append(row[col_index])
        else:
            # if the column identifier isn't a string or integer, it would not make sense
            raise ValueError("Invalid column identifier, please input a String or Integer value")

        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        row_num = 0
        for row in self.data:
            if row_num >= len(self.data): # breaks if we are at the maximum length of the table
                break
            for j in range(len(row)):
                try:
                    if row[j] == "": # if there is no value, don't do anything
                        pass
                    else:
                        numeric_val = float(row[j]) # converts a copy of the value to numeric
                        self.data[row_num][j] = numeric_val # appends numeric value into table
                except ValueError as e:
                    pass # if the value can't be made into a float, just continue
            row_num += 1
        pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        if isinstance(row_indexes_to_drop, int) == True: # if there is only 1 value to drop
            self.data.pop(row_indexes_to_drop)
        else: # if there is a list of values
            for i in reversed(row_indexes_to_drop): # pop them in reverse so we don't have an out of range error
                if i > len(self.data):
                    pass
                else:
                    self.data.pop(i)
        

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        
        infile = open(filename, "r") # opens file
        reader = csv.reader(infile)
        
        for row in reader:
            self.data.append(row) # appends table with rows from csv file
        self.column_names = self.data[0] # column names equal the header of the table
        self.drop_rows(0) # drop the header so we don't have to work around it every time
        self.convert_to_numeric() # converts all values to numeric to be used later

        infile.close() # close file

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w") # opens write file
        writer = csv.writer(outfile) 
        writer.writerow(self.column_names) # writes column names to the header of the file
        writer.writerows(self.data) # writes the rest of the table to the file
        outfile.close()

        

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns
            list of int: list of indexes of duplicate rows found
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        dupes = [] # stores both the duplicate and the original row index
        actual_dupes = [] # stores only the duplicate row index
        col_identifiers = [] # identifies the column indexes for the columns we are looking for
        for i in key_column_names:
            for j in range(len(self.column_names)):
                if i == self.column_names[j]:
                    col_identifiers.append(j) 
                else:
                    continue
                
        for k in range(len(self.data)): # searches through the data top down
            for l in reversed(range(len(self.data))): # searches through the data bottom up
                matched_columns = 0 # initalizes 0 columns matched at the start of search
                for m in col_identifiers: # looks through all column idenifier indexes
                    if k == l: # break if we are on the same row (just passing each other)
                        break
                    elif self.data[k][m] == self.data[l][m]: # if the values match up
                        matched_columns += 1
                        if matched_columns >= len(col_identifiers):
                            if [l, k] in dupes:
                                break # if we already have this dupe, pass by
                            else:
                                dupes.append([k,l]) # if it is a dupe, append it
                                if l not in actual_dupes: # if its not in the actual dupes, append it
                                    actual_dupes.append(l)
                            break
                        else:
                            continue
                    else:
                        break
        actual_dupes.sort() # sort it so it makes it usable
        return actual_dupes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        # this function doensn't necessarily "pop" NA values, but rather only adds the rows that don't
        #   have NA in them
        copy_table = []
        for row in self.data:
            range_of_val = 0
            for value in row:
                if value == "NA": # don't add row because it has NA
                    break # no reason to keep looking through row so we break
                elif range_of_val == len(row) -1 : # if we have reached the end of the length of the row, append
                    copy_table.append(row)
                else:
                    range_of_val += 1
        self.data = copy_table
        

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        copy_col = self.get_column(col_name, False) # copy of the original column but without missing values so we can calculate average
        col_avg = sum(copy_col) / len(copy_col)
        col_index = 0
        # finds the index of the column in self table so we can reference later
        for i in range(len(self.column_names)):
            if self.column_names[i] == col_name:
                col_index = i
        # if the indicated value is "NA", give it col_avg
        for i in range(len(self.data)):
            if self.data[i][col_index] == "NA":
                self.data[i][col_index] = col_avg
            else:
                pass
        
        

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column
        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        
        
        newTable = [] #creates an empty table with 6 columns for the stats and the amount of rows needed for the column names
        col_identifiers = []
        for i in col_names:
            for j in range(len(self.column_names)):
                if i == self.column_names[j]:
                    col_identifiers.append(j)
                else:
                    continue
                
        for i in range(len(col_identifiers)):
            row = [] # creates empty row to be filled with values related to the column
            curr_col = self.get_column(self.column_names[col_identifiers[i]], False) # gets the current column
            if curr_col == []: # if the column is blank, pass it
                continue
            else:
                row.append(self.column_names[col_identifiers[i]]) # attribute
                row.append(min(curr_col)) # min
                row.append(max(curr_col)) # max
                row.append((min(curr_col) + max(curr_col)) / 2) # mid
                row.append(sum(curr_col) / len(curr_col)) # avg
                curr_col.sort() # sorts all values to find the media
                if len(curr_col) % 2 == 0: # if there are an even amount of values
                    median = (curr_col[int((len(curr_col) - 1) / 2)] + curr_col[int(((len(curr_col) - 1) / 2) + 1)]) / 2 # find the average between the two values
                    row.append(median) # median
                else:
                    median_index = len(curr_col) // 2 # divide length in 2
                    row.append(curr_col[median_index]) # median
                newTable.append(row) # append all values

        summary_table_col_names = ["attribute", "min", "max", "mid", "avg", "median"] # gives summary table column names to match the rows
        summary_table = MyPyTable(summary_table_col_names, newTable)
        return summary_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        # creates a list of all the column names in the joined table
        joined_column_names = []
        for i in other_table.column_names:
            for j in self.column_names:
                if j not in joined_column_names:
                    joined_column_names.append(j)
            if i not in joined_column_names:
                joined_column_names.append(i)

        
        table = [] #creates empty table
        for left_row in self.data: # look through left table
            for right_row in other_table.data: # look through right table
                left_key = findKey(self, left_row, key_column_names) # finds left key (values we care about matching)
                right_key = findKey(other_table, right_row, key_column_names) # finds right key
                match = findMatch(left_key, right_key) # determines if they are a match
                temp_row = [] 
                if match == True: # if they do match
                    temp_row = joinMatchRows(left_row, right_row, self, other_table, joined_column_names) # join the rows
                    table.append(temp_row) # append the row of joined values
                else:
                    pass # don't do anything since we only care about matching vals
        
        
        inner_table = MyPyTable(joined_column_names, table) # create a new MyPyTable
        return inner_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """

        joined_column_names = [] # same as inner join
        for i in other_table.column_names:
            for j in self.column_names:
                if j not in joined_column_names:
                    joined_column_names.append(j)
            if i not in joined_column_names:
                joined_column_names.append(i)

        table = []
        for left_row in self.data:
            total_matches = 0 # initialize total matches at 0 every time we look through a row on the left table
            for right_row in other_table.data:
                # same as inner so far
                left_key = findKey(self, left_row, key_column_names)
                right_key = findKey(other_table, right_row, key_column_names)
                match = findMatch(left_key, right_key)
                temp_row = []
                if match == True:
                    total_matches += 1 # if they match, add 1 to total matches
                    temp_row = joinMatchRows(left_row, right_row, self, other_table, joined_column_names)
                    table.append(temp_row)
            
            if total_matches == 0: # if there are no matches in the left row
                temp_row = fillNA(left_row, self, joined_column_names) # fill the row with NA
                table.append(temp_row) # append it to join table
                
        outer_table = MyPyTable(joined_column_names, table) # creates MyPyTable here for easy reference to outer_table.column_names
        
        for left_row in other_table.data:
            total_matches = 0 # same as outer
            for right_row in outer_table.data:
                left_key = findKey(other_table, left_row, key_column_names)
                right_key = findKey(outer_table, right_row, key_column_names)
                match = findMatch(left_key, right_key)
                temp_row = []
                if match == True:
                    total_matches += 1 # this time we don't append because we already have the match
            if total_matches == 0: # we simply move onto this step
                temp_row = fillNA(left_row, other_table, joined_column_names) # same as first part but with the right table
                outer_table.data.append(temp_row) # appends with ".data" because it is a MyPyTable value now
        
        return outer_table


def findKey (table, row, key_column_names):
    '''Function that finds the key values and stores them in a list that is returned
        at the end of the function
        
        Args:
            table(MyPyTable): used for finding column indexes
            row(list of float and str): row of the table above
            key_column_names(list of str): list of all the key column names we want
            
        Returns:
            key(list of float and str): the key values needed to determine a match
    '''
    key = [] # initializes empty key
    for col_name in key_column_names:
        col_index = table.column_names.index(col_name)
        key.append(row[col_index]) # appends values where key column names are
    
    return key

def findMatch (left_key, right_key):
    '''Function that finds a match based on the keys given. Returns True if match is found
        and vice versa
    
        Args:
            left_key(list of str and float): values in the left row that are key to matching
            right_key(list of str and float): values in right row that are key to matching
            
        Return:
            match(bool): returns whether or not a match was found
    
    '''
    for i in left_key:
        if i in right_key:
            match = True # if all values in left key are in right key, it will ultimately return true
        else:
            match = False # if at any point the left key values are not in right key, it is returned false and we break
            break
    
    return match

def joinMatchRows(left_row, right_row, left_table, right_table, joined_col_names):
    '''Function that joins rows when they are matched
    
        Args:
            left_row(list of str and float): row of vals from left table
            right_row(list of str and float): row of vals from right table
            left_table(MyPyTable): left MyPyTable
            right_table(MyPyTable): right MyPyTable
            joined_cal_names(list of str): list of column names in join table
        
        Returns:
            joined_rows(list of str and float): contains combined vals from left_row and right_row
    
    '''
    # append left table first
    joined_rows = []
    for col_name in joined_col_names:
        # try: except: is used here in the case of a column name not being found in the table (same as below)
        try:
            col_index = left_table.column_names.index(col_name)
            joined_rows.append(left_row[col_index])
        except:
            pass
    # append right table next
    for col_name in joined_col_names:
        try:
            col_index = right_table.column_names.index(col_name)
            if right_row[col_index] in joined_rows:
                pass
            else:
                joined_rows.append(right_row[col_index])
        except:
            pass
        
    return joined_rows

def fillNA(row, table, joined_column_names):
    '''Fills row with the length of joined column names with "NA" when values are missing from given table
    
        Args:
            row (list of float and str): row usually found in the table
            table(MyPyTable): used to refer to the column names
            joined_column_names (list of str): list of the column names in the joined table
        
        Returns:
            NA_row(list of float and str): a row filled with "NA" vals when a value is not found 
                in the original table
    '''
    NA_row = []
    
    for col_name in joined_column_names:
        try:
            col_index = table.column_names.index(col_name)
            NA_row.append(row[col_index])
        except: # appends "NA" when the joined table column isn't found in the current table
            NA_row.append("NA")
            
    return NA_row