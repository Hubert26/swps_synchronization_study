# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:40:41 2024

@author: Hubert26
"""

#%%
import os
from pathlib import Path
import shutil


#%%
def create_directory(directory):
    """
    Creates a directory and all intermediate subdirectories if they do not exist.

    Args:
        directory (str): The path to the directory.

    Raises:
        OSError: If the directory cannot be created.
    """
    if not Path(directory).is_dir():
        try:
            # Create the directory, including any necessary intermediate directories
            os.makedirs(directory)
        except Exception as e:
            raise OSError(f"Failed to create directory: {e}")

#%%
def delete_directory(directory: str):
    """
    Deletes the specified directory and all its contents.

    Args:
        directory (str): The path to the directory to delete.

    Raises:
        FileNotFoundError: If the directory does not exist.
        OSError: If the directory or its contents cannot be deleted.
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    
    try:
        # Remove the directory and all its contents
        shutil.rmtree(directory_path)
    except Exception as e:
        raise OSError(f"Failed to delete directory: {e}")
        
#%%
def list_file_paths(directory, extension=None):
    """
    Retrieves a list of file paths from the specified directory.

    Args:
        directory (str): The path to the directory to search. Must be a valid directory path.
        extension (str, optional): The file extension to filter by (e.g., '.xlsx'). 
                                   If None, all files are returned.

    Returns:
        list of str: A list of file paths that match the specified extension.

    Raises:
        ValueError: If the provided `directory` is not a valid directory or does not exist.
    """
    dir_path = Path(directory)
    
    # Sprawdź, czy folder istnieje
    if not dir_path.is_dir():
        raise ValueError(f"The directory '{directory}' does not exist or is not a directory.")
    
    if extension:
        files = [str(file) for file in dir_path.rglob(f'*{extension}') if file.is_file()]
    else:
        files = [str(file) for file in dir_path.rglob('*') if file.is_file()]
    
    return files

#%%
def delete_file(file_path):
    """
    Deletes a file at the given path.

    Args:
        file_path (str): The path to the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be deleted.
    """
    # Check if the file exists
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    
    try:
        os.remove(file_path)
    except Exception as e:
        raise OSError(f"Failed to delete file: {e}")
        
#%%
def copy_file(src_path, dest_path):
    """
    Copies a file from the source path to the destination path.

    Args:
        src_path (str): The source file path.
        dest_path (str): The destination file path.

    Raises:
        FileNotFoundError: If the source file does not exist.
        OSError: If the file cannot be copied.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file does not exist: {src_path}")
    
    try:
        shutil.copy(src_path, dest_path)
    except Exception as e:
        raise OSError(f"Failed to copy file: {e}")

#%%
def read_text_file(file_path):
    """
    Reads the contents of a text file.

    Args:
        file_path (str): The path to the text file to read.

    Returns:
        str: The contents of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    # Check if the file exists
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            contents = file.read()
            return contents
    except Exception as e:
        raise OSError(f"Failed to read file: {e}")
        
#%%
def extract_file_name(file_path):
    """
    Extracts the file name from the given file path.

    Args:
        file_path (str): The full path to the file.

    Returns:
        str: The name of the file without the extension.
    """
    # Convert the file path to a Path object
    path = Path(file_path)
    
    # Extract the file name without extension
    file_name = path.stem
    
    return file_name

#%%
