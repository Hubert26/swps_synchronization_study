# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:27:45 2024

@author: Hubert Szewczyk
"""
import re


#%%
def count_words(input_string):
    """
    Count the number of words in the given string.

    Parameters:
    text (str): The input string to count words in.

    Returns:
    int: The number of words in the string. If the input is not a string, raises a ValueError.

    Raises:
        ValueError: If input_string is not a string
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string")
    
    words = input_string.split()
    return len(words)

#%%
def group_string_by_prefix(string_list, delimiter='_', **kwargs):
    """
    Groups strings in a list by the prefix (substring before the first occurrence of the delimiter).

    Args:
        string_list (list): List of strings to be grouped by their prefix.
        delimiter (str): The character or string by which the string will be split to determine the prefix.
        **kwargs: Optional keyword arguments for the split_string function, such as 'maxsplit'.

    Returns:
        dict: A dictionary where keys are the string prefixes and values are lists of strings sharing that prefix.
    """
    grouped_strings = {}

    for string in string_list:
        # Split the string using the delimiter and get the prefix
        prefix = string.split(delimiter, **kwargs)[0]
        
        # Group the strings by the prefix
        if prefix not in grouped_strings:
            grouped_strings[prefix] = []
        
        grouped_strings[prefix].append(string)
    
    return grouped_strings


#%%
def escape_html(input_string):
    """
    Escapes special HTML characters in a string to prevent HTML injection.

    This function replaces characters that have special meanings in HTML with their corresponding HTML escape codes. This is useful for ensuring that user input is displayed correctly on web pages without being interpreted as HTML.

    Args:
        input_string (str): The string to be escaped.

    Returns:
        str: The HTML-escaped string.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string.")
        
    # Replace special HTML characters with their escape codes
    return (input_string.replace("&", "&amp;")  # Replace ampersand
                .replace("<", "&lt;")      # Replace less-than sign
                .replace(">", "&gt;")      # Replace greater-than sign
                .replace('"', "&quot;")   # Replace double quote
                .replace("'", "&apos;"))  # Replace single quote


#%%
def remove_whitespace(input_string):
    """
    Removes all leading and trailing whitespace from the input string.

    Args:
        input_string (str): The string to be trimmed.

    Returns:
        str: The string without leading and trailing whitespaces.
    
    Raises:
        ValueError: If input_string is not a string or if delimiter is empty.
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string.")
        
    return input_string.strip()

#%%
def replace_substring(input_string, old_substring, new_substring):
    """
    Replaces occurrences of a substring with another substring.

    Args:
        input_string (str): The string where replacements will be made.
        old_substring (str): The substring to be replaced.
        new_substring (str): The substring to replace with.

    Returns:
        str: The string with replacements made.
    
    Raises:
        ValueError: If input_string is not a string or if delimiter is empty.
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string.")
        
    return input_string.replace(old_substring, new_substring)

#%%
def extract_numeric_suffix(input_string: str):
    """
    Extracts the numeric suffix from a given string.

    This function searches for a sequence of digits at the end of the input string. 
    If found, it returns the numeric suffix as an integer. If no numeric suffix is found, it returns None.

    Parameters:
    input_string (str): The input string from which the numeric suffix is to be extracted.

    Returns:
    int or None: The numeric suffix as an integer if found, otherwise None.
    
    Raises:
        ValueError: If input_string is not a string or if delimiter is empty.
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string.")
        
    match = re.search(r'\d+$', input_string)  # Search for digits at the end of the string
    if match:
        return int(match.group())  # Return the matched digits as an integer
    else:
        return None  # Return None if no digits are found

#%%
def extract_numeric_prefix(input_string: str):
    """
    Extracts the numeric prefix from a given string.

    This function searches for a sequence of digits at the beginning of the input string. 
    If found, it returns the numeric prefix as an integer. If no numeric prefix is found, it returns None.

    Parameters:
    input_string (str): The input string from which the numeric prefix is to be extracted.

    Returns:
    int or None: The numeric prefix as an integer if found, otherwise None.
    
    Raises:
        ValueError: If input_string is not a string.
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string.")
        
    match = re.search(r'^\d+', input_string)  # Search for digits at the beginning of the string
    if match:
        return int(match.group())  # Return the matched digits as an integer
    else:
        return None  # Return None if no digits are found
    
#%%
def remove_digits(input_string):
    """
    Removes all digits from the given string.

    Args:
        input_string (str): The string from which digits will be removed.

    Returns:
        str: The input string with all digits removed.
    """
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string.")
        
    return ''.join(filter(lambda x: not x.isdigit(), input_string))