import base64
import re


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_json_from_code_block(response: str):
    code_block = re.search(
            r'(?:```|~~~|、、、)(?:json\s*)?(.*?)(?:```|~~~|、、、)', 
            response, 
            re.DOTALL
        )
    
    json_str = code_block.group(1).strip()
    if json_str.endswith(','):
        json_str = json_str[:-1]
    
    return json_str

def extract_code_from_code_block(response: str):
    code_block = re.search(
            r'(?:```|~~~|、、、)(?:python\s*)?(.*?)(?:```|~~~|、、、)', 
            response, 
            re.DOTALL
        )
    python_str = code_block.group(1).strip()
   
    return python_str

def _assert_msgs(messages):
    for i, m in enumerate(messages):
        if not isinstance(m.content, (str, list)):
            raise TypeError(
                f"messages[{i}].content expects str|list, got {type(m.content)}: {repr(m.content)[:200]}"
            )

# 用法
# _assert_msgs(messages)
# resp = self.llm.invoke(messages).content.strip()
