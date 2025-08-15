from typing import List, Dict
import base64
import os
import json
import logging
import time
import re
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
from http import HTTPStatus
import tempfile


import dashscope
import requests
import google.generativeai as genai
import openai
from groq import Groq

logger = logging.getLogger("utilities")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encoded_img_to_pil_img(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image

def save_to_tmp_img_file(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path

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
def call_llm(engine, messages: List[Dict], VL_message: bool,  max_tokens=1000, top_p=0.9, temperature=0.5):
    model = engine.strip()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "temperature": temperature
    }
    load_dotenv()
    if model.startswith("azure-gpt-4o"):


        #.env config example :
        # AZURE_OPENAI_API_BASE=YOUR_API_BASE
        # AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT
        # AZURE_OPENAI_API_VERSION=YOUR_API_VERSION
        # AZURE_OPENAI_MODEL=gpt-4o-mini
        # AZURE_OPENAI_API_KEY={{YOUR_API_KEY}}
        # AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_API_BASE}/openai/deployments/${AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version=${AZURE_OPENAI_API_VERSION}


        # Load environment variables
        
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        #logger.info("Openai endpoint: %s", openai_endpoint)

        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        logger.info("Generating content with GPT model: %s", model)
        response = requests.post(
            openai_endpoint,
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            logger.error("Failed to call LLM: " + response.text)
            time.sleep(5)
            return ""
        else:
            return response.json()['choices'][0]['message']['content']
    elif model.startswith("gpt"):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        logger.info("Generating content with GPT model: %s", model)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            logger.error("Failed to call LLM: " + response.text)
            time.sleep(5)
            return ""
        else:
            return response.json()['choices'][0]['message']['content']

    elif model.startswith("claude"):
        claude_messages = []

        for i, message in enumerate(messages):
            claude_message = {
                "role": message["role"],
                "content": []
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"
            for part in message["content"]:

                if part['type'] == "image_url":
                    image_source = {}
                    image_source["type"] = "base64"
                    image_source["media_type"] = "image/png"
                    image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
                    claude_message['content'].append({"type": "image", "source": image_source})

                if part['type'] == "text":
                    claude_message['content'].append({"type": "text", "text": part['text']})

            claude_messages.append(claude_message)

        # the claude not support system message in our endpoint, so we concatenate it at the first user message
        if claude_messages[0]['role'] == "system":
            claude_system_message_item = claude_messages[0]['content'][0]
            claude_messages[1]['content'].insert(0, claude_system_message_item)
            claude_messages.pop(0)

        logger.debug("CLAUDE MESSAGE: %s", repr(claude_messages))

        headers = {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload["messages"] = claude_messages

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            logger.error("Failed to call LLM: " + response.text)
            time.sleep(5)
            return ""
        else:
            return response.json()['content'][0]['text']

    elif model.startswith("mistral"):
        assert VL_message is False, f"The model {model} can only support text-based input, please consider change based model or settings"

        mistral_messages = []

        for i, message in enumerate(messages):
            mistral_message = {
                "role": message["role"],
                "content": ""
            }

            for part in message["content"]:
                mistral_message['content'] = part['text'] if part['type'] == "text" else ""

            mistral_messages.append(mistral_message)

        from openai import OpenAI

        client = OpenAI(api_key=os.environ["TOGETHER_API_KEY"],
                        base_url='https://api.together.xyz',
                        )

        flag = 0
        while True:
            try:
                if flag > 20:
                    break
                logger.info("Generating content with model: %s", model)
                response = client.chat.completions.create(
                    messages=mistral_messages,
                    model=model,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature
                )
                break
            except:
                if flag == 0:
                    mistral_messages = [mistral_messages[0]] + mistral_messages[-1:]
                else:
                    mistral_messages[-1]["content"] = ' '.join(mistral_messages[-1]["content"].split()[:-500])
                flag = flag + 1

        try:
            return response.choices[0].message.content
        except Exception as e:
            print("Failed to call LLM: " + str(e))
            return ""

    elif model in ["gemini-pro", "gemini-pro-vision"]:
        if model == "gemini-pro":
            assert VL_message is False, f"The model {model} can only support text-based input, please consider change based model or settings"

        gemini_messages = []
        for i, message in enumerate(messages):
            role_mapping = {
                "assistant": "model",
                "user": "user",
                "system": "system"
            }
            gemini_message = {
                "role": role_mapping[message["role"]],
                "parts": []
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"

            # The gemini only support the last image as single image input
            if i == len(messages) - 1:
                for part in message["content"]:
                    gemini_message['parts'].append(part['text']) if part['type'] == "text" \
                        else gemini_message['parts'].append(encoded_img_to_pil_img(part['image_url']['url']))
            else:
                for part in message["content"]:
                    gemini_message['parts'].append(part['text']) if part['type'] == "text" else None

            gemini_messages.append(gemini_message)

        # the gemini not support system message in our endpoint, so we concatenate it at the first user message
        if gemini_messages[0]['role'] == "system":
            gemini_messages[1]['parts'][0] = gemini_messages[0]['parts'][0] + "\n" + gemini_messages[1]['parts'][0]
            gemini_messages.pop(0)

        # since the gemini-pro-vision donnot support multi-turn message
        if model == "gemini-pro-vision":
            message_history_str = ""
            for message in gemini_messages:
                message_history_str += "<|" + message['role'] + "|>\n" + message['parts'][0] + "\n"
            gemini_messages = [{"role": "user", "parts": [message_history_str, gemini_messages[-1]['parts'][1]]}]
            # gemini_messages[-1]['parts'][1].save("output.png", "PNG")

        # print(gemini_messages)
        api_key = os.environ.get("GENAI_API_KEY")
        assert api_key is not None, "Please set the GENAI_API_KEY environment variable"
        genai.configure(api_key=api_key)
        logger.info("Generating content with Gemini model: %s", model)
        request_options = {"timeout": 120}
        gemini_model = genai.GenerativeModel(model)

        response = gemini_model.generate_content(
            gemini_messages,
            generation_config={
                "candidate_count": 1,
                # "max_output_tokens": max_tokens,
                "top_p": top_p,
                "temperature": temperature
            },
            safety_settings={
                "harassment": "block_none",
                "hate": "block_none",
                "sex": "block_none",
                "danger": "block_none"
            },
            request_options=request_options
        )
        return response.text

    elif model.startswith("gemini"):
        gemini_messages = []
        for i, message in enumerate(messages):
            role_mapping = {
                "assistant": "model",
                "user": "user",
                "system": "system"
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"
            gemini_message = {
                "role": role_mapping[message["role"]],
                "parts": []
            }

            # The gemini only support the last image as single image input
            for part in message["content"]:

                if part['type'] == "image_url":
                    # Put the image at the beginning of the message
                    gemini_message['parts'].insert(0, encoded_img_to_pil_img(part['image_url']['url']))
                elif part['type'] == "text":
                    gemini_message['parts'].append(part['text'])
                else:
                    raise ValueError("Invalid content type: " + part['type'])

            gemini_messages.append(gemini_message)

        # the system message of gemini-1.5-pro-latest need to be inputted through model initialization parameter
        system_instruction = None
        if gemini_messages[0]['role'] == "system":
            system_instruction = gemini_messages[0]['parts'][0]
            gemini_messages.pop(0)

        api_key = os.environ.get("GENAI_API_KEY")
        assert api_key is not None, "Please set the GENAI_API_KEY environment variable"
        genai.configure(api_key=api_key)
        logger.info("Generating content with Gemini model: %s", model)
        request_options = {"timeout": 120}
        gemini_model = genai.GenerativeModel(
            model,
            system_instruction=system_instruction
        )

        with open("response.json", "w") as f:
            messages_to_save = []
            for message in gemini_messages:
                messages_to_save.append({
                    "role": message["role"],
                    "content": [part if isinstance(part, str) else "image" for part in message["parts"]]
                })
            json.dump(messages_to_save, f, indent=4)

        response = gemini_model.generate_content(
            gemini_messages,
            generation_config={
                "candidate_count": 1,
                # "max_output_tokens": max_tokens,
                "top_p": top_p,
                "temperature": temperature
            },
            safety_settings={
                "harassment": "block_none",
                "hate": "block_none",
                "sex": "block_none",
                "danger": "block_none"
            },
            request_options=request_options
        )

        return response.text

    elif model == "llama3-70b":

        assert VL_message is False, f"The model {model} can only support text-based input, please consider change based model or settings"

        groq_messages = []

        for i, message in enumerate(messages):
            groq_message = {
                "role": message["role"],
                "content": ""
            }

            for part in message["content"]:
                groq_message['content'] = part['text'] if part['type'] == "text" else ""

            groq_messages.append(groq_message)

        # The implementation based on Groq API
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        flag = 0
        while True:
            try:
                if flag > 20:
                    break
                logger.info("Generating content with model: %s", model)
                response = client.chat.completions.create(
                    messages=groq_messages,
                    model="llama3-70b-8192",
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature
                )
                break
            except:
                if flag == 0:
                    groq_messages = [groq_messages[0]] + groq_messages[-1:]
                else:
                    groq_messages[-1]["content"] = ' '.join(groq_messages[-1]["content"].split()[:-500])
                flag = flag + 1

        try:
            return response.choices[0].message.content
        except Exception as e:
            print("Failed to call LLM: " + str(e))
            return ""

    elif model.startswith("qwen"):
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        qwen_messages = []

        for i, message in enumerate(messages):
            qwen_message = {
                "role": message["role"],
                "content": []
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"
            for part in message["content"]:
                qwen_message['content'].append(
                    {"image": "file://" + save_to_tmp_img_file(part['image_url']['url'])}) if part[
                                                                                                    'type'] == "image_url" else None
                qwen_message['content'].append({"text": part['text']}) if part['type'] == "text" else None

            qwen_messages.append(qwen_message)

        flag = 0
        while True:
            try:
                if flag > 20:
                    break
                logger.info("Generating content with model: %s", model)

                if model in ["qwen-vl-plus", "qwen-vl-max", "qwen-vl-max-2025-04-08", "qwen-vl-max-2025-01-25"]:
                    response = dashscope.MultiModalConversation.call(
                        model=model,
                        messages=qwen_messages,
                        result_format="message",
                        max_length=max_tokens,
                        top_p=top_p,
                        temperature=temperature
                    )

                elif model in ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-max-0428", "qwen-max-0403",
                                    "qwen-max-0107", "qwen-max-longcontext"]:
                    response = dashscope.Generation.call(
                        model=model,
                        messages=qwen_messages,
                        result_format="message",
                        max_length=max_tokens,
                        top_p=top_p,
                        temperature=temperature
                    )

                else:
                    raise ValueError("Invalid model: " + model)

                if response.status_code == HTTPStatus.OK:
                    break
                else:
                    logger.error('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                        response.request_id, response.status_code,
                        response.code, response.message
                    ))
                    raise Exception("Failed to call LLM: " + response.message)
            except:
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded, retrying...")
                    time.sleep(10)
                flag = flag + 1

        try:
            if model in ["qwen-vl-plus", "qwen-vl-max", "qwen-vl-max-2025-04-08", "qwen-vl-max-2025-01-25"]:
                return response['output']['choices'][0]['message']['content'][0]['text']
            else:
                return response['output']['choices'][0]['message']['content']

        except Exception as e:
            print("Failed to call LLM: " + str(e))
            return ""
    else:
        raise ValueError("Invalid model: " + model)
