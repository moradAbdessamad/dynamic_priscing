from importlib import resources
import json

def load_model_metadata():
    """
    Return the content of metaData/modelMetaData.json as a dict.
    """
    with resources.open_text(__package__ + ".metaData", "modelMetaData.json") as fp:
        return json.load(fp)
