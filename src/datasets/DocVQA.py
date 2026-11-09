# Imports
import zipfile
import torch
import pandas as pd
import os
import json
import tarfile
from dataclasses import dataclass, field
import torchvision.transforms as transforms
from typing import List, Tuple


# Constants 
# File names for test, validation and train for annotation
validation_file_name = "val_v1.0_withQT.json"
train_file_name = "train_v1.0_withQT.json"
test_file_name = "test_v1.0.json"


def get_annotation_from_zip(path:str, split:str ="train"):
    """Get question and answer annotations from file.

    The dataframe with questions has the following columns: 
    - questionId, question, question_types, docId, ucsf_document_id, ucsf_document_page_no, answers, document_id

    Args:
        path (str): Absolute path to the zipped annotation folder 
        split (str, optional): Type of dataset. Only loads the annotation for the given dataset split. Defaults to "train".

    Raises:
        ValueError: Raised if "split" is not set to train, val, or test. 

    Returns:
        pd.DataFrame: Dataframe with the annotated question and answer data
    """

    # Open the zipped annotated file
    with zipfile.ZipFile(path, 'r') as zf:
        # Read each data split to its own dataframe 
        if split == "train":
            with zf.open(train_file_name) as file:
                df = pd.read_json(file)
        elif split == "val":
            with zf.open(validation_file_name) as file:
                df = pd.read_json(file)
        elif split == "test":
            with zf.open(test_file_name) as file:
                df = pd.read_json(file)
        else:
            # Raise error if the split was not valid
            raise ValueError("Split was not train, val or test")
    
    # print(f"Columns after read: {df.columns}")

    # Assert that all values within the column are the current data split
    assert (df["dataset_split"] == split).all(), f"Not all values in the 'data_split' column are '{split}'"

    # Then remove the data split column
    df = df.drop(columns=["dataset_split"])

    # Transform the data into a new dataframe from the "data" column
    # This is were the information are 
    df = pd.json_normalize(df["data"])
        
    # Replace "documents/" with "" and ".png" with "" to create the 'document_id'
    df["document_id"] = df["image"].str.replace("documents/", "").str.replace(".png", "")

    # Remove the 'image' column
    df = df.drop(columns=["image"])

    # Return the dataframe
    return df



def get_split_from_path(path: str) -> str:
    """Get the type of split from the path to the annotated json file

    Args:
        path (str): path to the annotation file. 

    Returns:
        str: "train", "test" or "val" based on the json file provided.
    """
    parts = path.split("/")
    last_part = parts[-1].replace(".json", "")
    return last_part.split("_")[0]


def get_ocr_data_from_zip(path: str) -> pd.DataFrame:
    """Get OCR dataframe 

    The dataframe has the columns:
    - document_id, recognitionResults, status 

    Args:
        path (str): Absolute path to the provided .zip file with annotations 

    Returns:
        pd.DataFrame: dataframe with OCR data 
    """
    # Initialize an empty list to hold OCR data
    ocr_data = []

    # Open the .tar.gz file
    with tarfile.open(path, 'r:gz') as tar:
        # List the contents of the .tar.gz archive
        ocr_members = tar.getmembers()

        # Get only files 
        ocr_files = [ocr_member for ocr_member in ocr_members if ocr_member.isfile()]

        # Iterate over the files
        for file_name in ocr_files:            
            # Open the JSON file from the archive
            with tar.extractfile(file_name) as file:
                # Read and load the JSON data from the file
                ocr_item_json = json.load(file)
                
                # Normalize the JSON data on recognition results
                ocr_item_df = pd.json_normalize(ocr_item_json["recognitionResults"])
                
                # Add the filename to the dataframe
                document_id = file_name.name.replace(".json", "")
                ocr_item_df['document_id'] = document_id

                # Append the DataFrame to the list
                ocr_data.append(ocr_item_df)

    # Concatenate all DataFrames at once
    ocr_df = pd.concat(ocr_data, ignore_index=True)

    # Return OCR dataframe
    return ocr_df
    

def get_image_data_from_zip(path:str) -> pd.DataFrame:
    """Get image dataframe

    The dataframe has the following columns:
    - document_id, image_path

    Args:
        path (str): Absolute path to the zipped folder with all the images

    Returns:
        pd.DataFrame: dataframe with image information
    """
    # Open the .tar.gz file
    with tarfile.open(path, 'r:gz') as tar:
        # List the contents of the .tar.gz archive
        image_members = tar.getmembers()

    # Only files 
    image_paths = [image_member for image_member in image_members if image_member.isfile()]

    # Image items
    image_items = []

    # Create an image item for image with document id and the full path to the image
    for current_image_item in image_paths:
        # Name of the file 
        document_id = current_image_item.name.replace(".png", "")

        # Absolute path to the image
        image_abs_path = os.path.abspath(os.path.join(path, current_image_item.name))

        # Append the item to the list
        item = [document_id, image_abs_path]
        image_items.append(item)

    # Make a dataframe and return it 
    image_df = pd.DataFrame(image_items, columns=['document_id', 'image_path'])
    return image_df


@dataclass
class OcrLine:
    boundingBox: List[int]
    text: str

@dataclass
class OcrInfo:
    page: str
    clockwiseOrientation: str
    width: str
    height: str
    unit: str
    lines: List[OcrLine]

@dataclass
class DocumentData:
    # Image information 
    document_id: str
    image_path: str 

    # Question and answer information
    question: str
    question_types: List[str]
    question_id: int
    answers: List[str]

    # OCR info for the given image
    ocr_data: OcrInfo


class DocVQADataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file_path:str, image_data_file_path:str, ocr_file_path:str, split:str = "train", verbose: bool = False):
        # Assert that the given paths exists 
        for path in [annotations_file_path, image_data_file_path, ocr_file_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} does not exist")


        # Print the paths received
        if verbose:
            print("Annotation path:", annotations_file_path)
            print("Image data path:", image_data_file_path)
            print("OCR file path:", ocr_file_path)
        

        # Store paths to files 
        self.annotations_file_path = annotations_file_path
        self.image_data_file_path = image_data_file_path
        self.ocr_file_path = ocr_file_path

        # Get annotations data
        self.annotation_data_df = get_annotation_from_zip(self.annotations_file_path, split=split)
        if verbose:
            print("Loaded annotation dataframe")

        # Get OCR data as JSON 
        self.ocr_df = get_ocr_data_from_zip(self.ocr_file_path)
        if verbose:
            print("Loaded OCR dataframe")

        # Get image dataframe 
        self.image_df = get_image_data_from_zip(self.image_data_file_path)
        if verbose:
            print("Loaded image dataframe")

        # For each question annotation create document data 
        self.document_data = []

        if verbose: 
            print("Start processing document items")
        
        # Iterate over all annotations and and create the data for the annotation
        for _, row in self.annotation_data_df.iterrows():
            # Extract document_id from the annotation row
            doc_id = row["document_id"]

            # Look up the image path for this document
            image_row = self.image_df[self.image_df["document_id"] == doc_id]
            if image_row.empty:
                if verbose:
                    print(f"No image found for document_id: {doc_id}")
                break
            image_path = image_row.iloc[0]["image_path"]

            # Look up OCR information for this document
            ocr_row = self.ocr_df[self.ocr_df["document_id"] == doc_id]
            if not ocr_row.empty:
                ocr_item = ocr_row.iloc[0].to_dict()
                # Convert OCR lines into a list of OcrLine objects
                ocr_lines = []
                for line in ocr_item.get("lines", []):
                    ocr_lines.append(OcrLine(
                        boundingBox=line.get("boundingBox", []),
                        text=line.get("text", "")
                    ))

                ocr_info = OcrInfo(
                    page=ocr_item.get("page", ""),
                    clockwiseOrientation=ocr_item.get("clockwiseOrientation", ""),
                    width=ocr_item.get("width", ""),
                    height=ocr_item.get("height", ""),
                    unit=ocr_item.get("unit", ""),
                    lines=ocr_lines
                )
            else:
                # If no OCR data is found, create a default empty OCR info
                ocr_info = OcrInfo(page="", clockwiseOrientation="", width="", height="", unit="", lines=[])
                if verbose:
                    print(f"No OCR data was found for document {doc_id}")

            # Create a DocumentData instance for the current annotation
            document_data_item = DocumentData(
                document_id=doc_id,
                image_path=image_path,
                question=row.get("question", ""),
                question_types=row.get("question_types", []),
                question_id=row.get("questionId", 0),
                answers=row.get("answers", []),
                ocr_data=ocr_info
            )

            # Add to the document data list
            self.document_data.append(document_data_item)

        if verbose:
            print(f"Created document data for {len(self.document_data)} annotations.")
         
    
    def __len__(self):
        # Return how many annotations we have 
        return len(self.document_data)

    def __getitem__(self, idx):
        document_item = self.document_data[idx]
        return document_item