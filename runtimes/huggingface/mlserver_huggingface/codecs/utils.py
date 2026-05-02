import io
import json
from typing import Any
import base64
import numpy as np
from PIL import Image, ImageChops
from transformers.pipelines import Conversation
from mlserver.codecs.json import JSONEncoderWithArray

IMAGE_PREFIX = "data:image/"
DEFAULT_IMAGE_FORMAT = "PNG"


class HuggingfaceJSONEncoder(JSONEncoderWithArray):
    """Custom JSON encoder for Hugging Face objects.
    
    Extends JSONEncoderWithArray to handle serialization of PIL Images and
    Transformers Conversation objects. Images are encoded as base64 data URIs,
    and Conversations are serialized to dictionaries with their key attributes.
    """
    
    def default(self, obj):
        """Serialize custom objects to JSON-compatible formats.
        
        Args:
            obj: The object to serialize. Supports PIL Image.Image and
                 transformers.pipelines.Conversation objects.
        
        Returns:
            JSON-serializable representation of the object. Images become
            base64-encoded data URIs, Conversations become dictionaries.
            Falls back to parent class for unsupported types.
        """
        # Handle PIL Image objects by converting to base64-encoded data URI
        if isinstance(obj, Image.Image):
            buf = io.BytesIO()
            # Ensure image has a format; default to PNG if not set
            if not obj.format:
                obj.format = DEFAULT_IMAGE_FORMAT
            obj.save(buf, format=obj.format)
            # Build data URI: data:image/<format>;base64,<encoded_data>
            return (
                IMAGE_PREFIX
                + obj.format.lower()
                + ";base64,"
                + base64.b64encode(buf.getvalue()).decode()
            )
        # Handle Conversation objects by extracting key attributes
        elif isinstance(obj, Conversation):
            return {
                "uuid": str(obj.uuid),
                "past_user_inputs": obj.past_user_inputs,
                "generated_responses": obj.generated_responses,
                "new_user_input": obj.new_user_input,
            }
        else:
            return super().default(obj)


def json_encode(payload: Any, use_bytes: bool = False):
    if use_bytes:
        return json.dumps(payload, cls=HuggingfaceJSONEncoder).encode()
    return json.dumps(payload, cls=HuggingfaceJSONEncoder)


def json_decode(payload):
    raw_dict = json.loads(payload)
    return Convertor.do(raw_dict)


conversation_keys = {
    "uuid",
    "past_user_inputs",
    "generated_responses",
    "new_user_input",
}


class Convertor:
    """Converts JSON dictionaries back to their original object types.
    
    Recursively processes JSON data to reconstruct PIL Images from base64
    data URIs and Conversation objects from their serialized dictionaries.
    Handles nested structures including lists and dictionaries.
    """
    
    @classmethod
    def do(cls, raw):
        """Recursively convert JSON data to original object types.
        
        Args:
            raw: JSON-decoded data (dict, list, or primitive value).
        
        Returns:
            Converted data with PIL Images and Conversations reconstructed.
        """
        if isinstance(raw, dict):
            return cls.convert_dict(raw)
        elif isinstance(raw, list):
            return cls.convert_list(raw)
        else:
            return raw

    @classmethod
    def convert_conversation(cls, d: dict[str, Any]):
        """Reconstruct a Conversation object from a dictionary.
        
        Args:
            d: Dictionary potentially containing Conversation fields.
        
        Returns:
            Conversation object if dict matches conversation_keys schema,
            None otherwise.
        """
        if set(d.keys()) == conversation_keys:
            return Conversation(
                text=d["new_user_input"],
                conversation_id=d["uuid"],
                past_user_inputs=d["past_user_inputs"],
                generated_responses=d["generated_responses"],
            )
        return None

    @classmethod
    def convert_dict(cls, d: dict[str, Any]):
        """Recursively convert dictionary values to original object types.
        
        Args:
            d: Dictionary with values that may need conversion.
        
        Returns:
            Dictionary with PIL Images, Conversations, and nested structures
            reconstructed from their serialized forms.
        """
        # Check if this entire dict is a serialized Conversation
        conversation = cls.convert_conversation(d)
        if conversation is not None:
            return conversation
        tmp = {}
        for k, v in d.items():
            # Recursively process nested dictionaries
            if isinstance(v, dict):
                # Check if nested dict is a Conversation
                if set(v.keys()) == conversation_keys:
                    tmp[k] = Conversation(text=v["new_user_input"])
                else:
                    tmp[k] = cls.convert_dict(v)
            # Recursively process lists
            elif isinstance(v, list):
                tmp[k] = cls.convert_list(v)
            # Decode base64-encoded image data URIs back to PIL Images
            elif isinstance(v, str):
                # Check for data:image/... URI and decode
                if v.startswith(IMAGE_PREFIX):
                    decoded = base64.b64decode(v.split(",")[1])
                    buf = io.BytesIO(decoded)
                    tmp[k] = Image.open(buf)
                else:
                    tmp[k] = v  # type: ignore
            else:
                tmp[k] = v
        return tmp

    @classmethod
    def convert_list(cls, list_data: list[Any]):
        """Recursively convert list elements to original object types.
        
        Args:
            list_data: List potentially containing serialized objects.
        
        Returns:
            List with elements converted to their original types.
        """
        nl = []
        for el in list_data:
            if isinstance(el, list):
                nl.append(cls.convert_list(el))
            elif isinstance(el, dict):
                nl.append(cls.convert_dict(el))
            else:
                nl.append(el)
        return nl


class EqualUtil:
    """Utility for deep equality comparison of complex data structures.
    
    Provides methods to compare dictionaries and lists that may contain
    PIL Images, numpy arrays, and nested structures. Uses pixel-level
    comparison for images and array_equal for numpy arrays.
    """
    
    @staticmethod
    def pil_equal(img1: "Image.Image", img2: "Image.Image") -> bool:
        """Compare two PIL images for pixel-level equality.
        
        Args:
            img1: First PIL Image.
            img2: Second PIL Image.
        
        Returns:
            True if images are identical, False otherwise.
        """
        diff = ImageChops.difference(img1, img2)
        if diff.getbbox() is None:
            return True
        return False

    @staticmethod
    def _compare_elements(el1: Any, el2: Any) -> bool:
        """Compare two elements with type-specific equality checks.
        
        Args:
            el1: First element.
            el2: Second element.
        
        Returns:
            True if elements are equal, False otherwise.
        """
        if isinstance(el1, dict):
            return EqualUtil.dict_equal(el1, el2)
        elif isinstance(el1, list):
            return EqualUtil.list_equal(el1, el2)
        elif isinstance(el1, Image.Image):
            return EqualUtil.pil_equal(el1, el2)
        elif isinstance(el1, np.ndarray):
            return np.array_equal(el1, el2)
        else:
            return el1 == el2
    
    @staticmethod
    def list_equal(list1: list[Any], list2: list[Any]) -> bool:
        """Perform deep equality comparison of two lists.
        
        Handles lists containing dictionaries, nested lists, PIL Images,
        numpy arrays, and primitive values. Uses type-specific comparison
        methods for complex types.
        
        Args:
            list1: First list to compare.
            list2: Second list to compare.
        
        Returns:
            True if lists are deeply equal, False otherwise.
        """
        # Early return for length mismatch
        if len(list1) != len(list2):
            return False
        # Compare each element using type-specific comparison
        for idx, el in enumerate(list1):
            if not EqualUtil._compare_elements(el, list2[idx]):
                return False
        return True

    @staticmethod
    def dict_equal(dict1: dict[Any, Any], dict2: dict[Any, Any]) -> bool:
        """Perform deep equality comparison of two dictionaries.
        
        Handles dictionaries containing PIL Images, numpy arrays, nested
        dictionaries, lists, and primitive values. Uses type-specific
        comparison methods for complex types.
        
        Args:
            dict1: First dictionary to compare.
            dict2: Second dictionary to compare.
        
        Returns:
            True if dictionaries are deeply equal, False otherwise.
        """
        # Early return for key set mismatch
        if not set(dict1.keys()) == set(dict2.keys()):
            return False
        # Compare each value using type-specific comparison
        for k, v in dict1.items():
            if not EqualUtil._compare_elements(v, dict2[k]):
                return False
        return True
