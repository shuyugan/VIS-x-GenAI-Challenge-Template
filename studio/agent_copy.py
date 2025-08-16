
import csv
import os
import pandas as pd
import json

from typing import Any, Dict, List
from helpers import get_llm
from tabulate import tabulate
from report_html import generate_html_report
from report_pdf import generate_pdf_report
from langchain_core.messages import HumanMessage, SystemMessage

from prompts import CollectorPrompt, VisualizerPrompt, InsightPrompt
from utils import extract_json_from_code_block, extract_code_from_code_block, encode_image
class Agent:
    def __init__(self):
        self.llm = get_llm(temperature=0, max_tokens=4096)
        self.sample_size = 10
        self.num_directions = 10
        self.metadata_report = None

        self.gen_explanation_prompt = CollectorPrompt.GEN_EXPLANATION_SYS_PROMPT
        self.direction_advisor_prompt = VisualizerPrompt.DIRECTION_ADVISOR
        self.code_generator_prompt = VisualizerPrompt.CODE_GENERATOR
        self.code_rectifier_prompt = VisualizerPrompt.CODE_RECTIFIER
        self.generate_insight_prompt = InsightPrompt.GEN_INSIGHT
        self.verify_insight_prompt = InsightPrompt.EVALUATE_INSIGHT
        self.check_image_prompt = VisualizerPrompt.CHART_QUALITY_CHECKER
    def initialize(self):
        self.data_path = './dataset.csv'
        self.dataset = pd.read_csv(self.data_path)
        self.sampled_data = self.dataset.sample(n=self.sample_size).fillna("NULL")
        print(f"Loaded dataset with shape: {self.dataset.shape}")

    def generate_metadatareport(self) -> Dict[str, Any]:
        df = self.dataset
        sample_table = self.sampled_data.to_markdown(index=False)
        dtype_info = "\n".join([f"- `{col}`: {str(dtype)}" 
                          for col, dtype in df.dtypes.items()])
        messages = []
        
        metadata = {
            "shape": f"{len(df)} rows * {len(df.columns)} columns",
            "dtypes": dtype_info,
        }

        messages.append(SystemMessage(content=self.gen_explanation_prompt))
        
        sniff_info = f"""
            ### Dataset Metadata
            {json.dumps(metadata, indent=2)}

            ### Sample Data (First {self.sample_size} Rows)
            {sample_table}
        """
        messages.append(
            HumanMessage(content="Here is the sniffed information, including dataset metadata and sample data")
        )
        messages.append(
            HumanMessage(content=sniff_info)
        )
        print("Generating dataset introduction...")
        introduction = self.llm.invoke(messages).content.strip()
        print(f"Introduction: {introduction}")
        
        metadata["introduction"] = introduction
        metadata["sample_data"] = sample_table

        return metadata
    
    def direction_advisor(self) -> str:
        prompt = self.direction_advisor_prompt.replace("{num_directions}", str(self.num_directions))
        metadata_info = json.dumps(self.metadata_report, indent=2)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=metadata_info)
        ]
        print("Advising visualization directions...")
        response = self.llm.invoke(messages).content.strip()
        json_str = extract_json_from_code_block(response)
        print(f"Response: {json_str}")
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"in direction_advisor: Invalid JSON format: {str(e)}")
    
    def code_rectify(self, code: str, error: str) -> str:
        prompt = self.code_rectifier_prompt
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"""Original Code: {code}
                                    Error: {error}""")
        ]

        response = self.llm.invoke(messages).content.strip()
        rectified_code = extract_code_from_code_block(response)

        return rectified_code
    
    def execute_code(self, code: str):
        flag, cnt = True, 0
        while flag and cnt < 5:
            with open("temp_code.py", "w", encoding='utf-8') as f:
                f.write(code)
    
            try:
                exec(code)
                print("Code executed successfully.")
                flag = False
            except Exception as e:
                error = str(e)
                print(f"Invalid code: {error}")
                cnt += 1
                code = self.code_rectify(code, error)
        if flag:
            raise RuntimeError(f"Failed to execute code after multiple attempts: {error}")

    def generate_plot_code(self, direction: Dict, idx: int) -> str:
        prompt = self.code_generator_prompt.format(
            data_path = self.data_path,
            output_path = f"plot_{idx}.png",
        )
        metadata_info = json.dumps(self.metadata_report, indent=2)
        payload = f"""
            Metadata Information:
            {metadata_info}

            Direction:
            {json.dumps(direction, indent=2)}
        """

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=payload)
        ]
        print(f"Generating plot code for direction {idx}...")
        response = self.llm.invoke(messages).content.strip()
        code = extract_code_from_code_block(response)
        print(f"Generated code:\n{code}")

        return code
    def check_image_quality(self, image_lst: List[str]) -> List[str]:
        verified_images = []
        for image_path in image_lst:
            base64_image = encode_image(image_path)
            messages = [
                SystemMessage(content=self.check_image_prompt),
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": "Please check the quality of the following chart image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ])
            ]
            print(f"Checking quality for image: {image_path}")
            response = self.llm.invoke(messages).content.strip()
            quality = extract_json_from_code_block(response)

            try:
                data = json.loads(quality)
                is_legible = data["is_legible"]
            except:
                is_legible = False
            
            if is_legible:
                verified_images.append(image_path)
            
            print(f"Image {image_path} legibility: {is_legible}")
        
        return verified_images

    def generate_insight(self, image_path: str) -> Dict:
        base64_image = encode_image(image_path)

        cnt = 0

        while cnt < 5:

            messages = [
                SystemMessage(content=self.generate_insight_prompt),
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": "Given the plot as below. What's the insight you can get from it?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ])
            ]
            print(f"Generating insight for image: {image_path}")
            response = self.llm.invoke(messages).content.strip()

            insight_content = extract_json_from_code_block(response)
            print(f"Insight response: {insight_content}")

            try:
                data = json.loads(insight_content)
                if not isinstance(data.get("insights"), list):
                    print("Invalid structure: 'insights' must be a list")
                    cnt += 1
                    continue          
                return data
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error decoding insight for {image_path}: {e}")
                cnt += 1
                continue
        
        if cnt == 5:
            raise ValueError(f"Failed to generate valid insight after multiple attempts for image: {image_path}")
    def evaluate_insight(self, insight_dict: Dict) -> Dict:
        final_insight_dict = {img_path: [] for img_path in insight_dict.keys()}

        metadata_info = json.dumps({"shape": self.metadata_report["shape"], "dtypes": self.metadata_report["dtypes"],}, indent=2)

        for img_path, insights in insight_dict.items():
            print(f"Evaluating insights for image: {img_path}")

            for insight in insights:
                messages = [
                    SystemMessage(content=self.verify_insight_prompt),
                    HumanMessage(content=[
                            {
                                "type": "text",
                                "text": f"Here is the Insight: {insight}\n\nAlso, here is the Metadata Information of the dataset:\n\n{metadata_info}"
                            },

                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(img_path)}",
                                    "detail": "high"
                                }

                            }

                        ]
                    )
                ]

                response = self.llm.invoke(messages).content.strip()
                json_content = extract_json_from_code_block(response)

                try:
                    data = json.loads(json_content)
                    final_insight_dict[img_path].append(data)
                except Exception as e:
                    raise ValueError(f"in evaluate_insight: Error parsing JSON response for image {img_path} and insight {insight}: {str(e)}")

        return final_insight_dict
    
    def present_report(self, final_insight_dict: Dict):
        ...
    def process(self):
        print("Beginning processing...")
        self.metadata_report = self.generate_metadatareport()

        directions = self.direction_advisor()

        img_files = []

        for i, direction in enumerate(directions):
            plot_code = self.generate_plot_code(direction, idx=i+1)
            self.execute_code(plot_code)
            img_files.append(f"plot_{i+1}.png")
        
        verified_img_files = self.check_image_quality(img_files)

        print(f"Here is the verified images: {verified_img_files}")
        
        insight_dict = {img_file: [] for img_file in verified_img_files}

        for img_file in verified_img_files:
            image_path = img_file
            data = self.generate_insight(image_path)

            insight_dict[img_file] = [desc['description'] for desc in data["insights"]]
        
        final_insight_dict = self.evaluate_insight(insight_dict)

        with open("insights_evaluated.json", "w", encoding="utf-8") as f:
            json.dump(final_insight_dict, f, ensure_ascii=False, indent=4)
        
        with open("insights.json", "w", encoding="utf-8") as f:
            json.dump(insight_dict, f, ensure_ascii=False, indent=4)

        

        





        
