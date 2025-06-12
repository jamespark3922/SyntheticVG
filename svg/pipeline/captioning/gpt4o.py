import ast
from PIL import Image
import logging

from . import ObjectCaptioner
from svg.openai_utils import OpenaiAPI

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

SYS_PROMPT2="""
Provide objects that are easily groundable and visibly present in the image.
Avoid abstract objects or conepts that are not easily groundable. 
The goal is to run a grounding model on the objects you provide to generate bounding boxes and segmentation regions for objects in the image.
To do so, first provide a dense description of the image detailing every significant objects and their relations or interactions it has with other objects.
Then, give a list of objects that I should include definitely include so that I can ground and provide bounding boxes for them.
In your object list, provide only the basic noun phrases of the objects and avoid using any adjectives or verbs (you can include color, or visually distinctive elements).
If there are multiple common objects, describe with only the most common object name, instead of identifying each object separately.
"""

SYS_PROMPT="""
Provide a dense description of the image detailing every significant objects and their relations or interactions it has with other objects.
Try to segment the different events and parts in the image and provide a detailed description of each event.
Lastly, give a list of objects that I should include definitely include so that I can ground and provide bounding boxes for them. 
Do not mention anything from the prompt in your response.
"""

class GPT4Captioner(ObjectCaptioner):
    def __init__(self, api_key=None):

        # Wrapper for openai API calls
        self.openai_api = OpenaiAPI(api_key=api_key)

    def generate(
        self, 
        image: Image.Image, 
        gpt_model: str = 'gpt-4o-2024-08-06',
        temperature: float = 0.5,
        max_tokens: int = 2048
    ) -> dict:
        json_format = """ 
        {
            'text': only the dense text paragraph of the image.
            'objects': list of all objects that should be included from your description.
        }
    """
        response = self.openai_api.call_chatgpt(
            model=gpt_model, # 'gpt-4o-2024-08-06',
            sys_prompt=SYS_PROMPT2,
            usr_prompt=f"Provide your response for this image in JSON. {json_format}",
            image_input=image,
            response_format='json',
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response is None:
            return None
        gpt_result, usage = response    

        for _ in range(2):
            try:
                gpt_result = ast.literal_eval(gpt_result)
                caption: str = gpt_result['text']
                objects: list[str] = gpt_result['objects']

                result = {
                    'caption': caption,
                    'objects': objects,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens,
                }

                return result
            except Exception as e:
                logging.error(f"Error in parsing GPT response: {e}")
                logging.error(f"Response: {gpt_result}")
        return None
    
    def generate_objects(self, image) -> list[str]:
        result = self.generate(image)
        if result is None:
            return None
        else:
            return result['objects']