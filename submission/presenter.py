import json
import re
import os
from typing import Any, Dict, List


from prompts import PresenterPrompt
from utils import extract_json_from_code_block, encode_image, _assert_msgs
from langchain_core.messages import HumanMessage, SystemMessage

from md_pdf import markdown_to_pdf

from datetime import date
class Presenter:
    def __init__(self, llm, metadata_report, visualize_directions, verified_images_files, final_insight):
        self.llm = llm
        self.metadata_report = metadata_report
        self.introduction = metadata_report['introduction']
        self.visualize_directions = visualize_directions
        self.verified_images_files = verified_images_files
        self.final_insight = final_insight

        self.gen_introduction_prompt = PresenterPrompt.INTRODUCTION_GENERATOR
        self.gen_narrative_prompt = PresenterPrompt.NARRATIVE_COMPOSER
        self.order_generator_prompt = PresenterPrompt.ORDER_GENERATOR
        self.transition_generator_prompt = PresenterPrompt.TRANSITION_GENERATOR
        self.conclusion_generator_prompt = PresenterPrompt.CONCLUSION_GENERATOR
        self.judger_prompt = PresenterPrompt.JUDGER
    
    def initialize(self):
        self.topic_lst = []
        print(f"Verified images: {str(self.verified_images_files)}")
        for image_name in self.verified_images_files:
            idx = int(image_name.split('_')[-1].split('.')[0]) - 1
            topic = self.visualize_directions[idx]['topic'] + ': ' + self.visualize_directions[idx]['explanation']
            self.topic_lst.append(topic)

    def generate_order(self):
        print('Generating order...')
        topics_block = "\n".join(self.topic_lst)
        content = f"Here is the brief introduction of the dataset: {self.introduction}\n\n, And here is the list of topics to be covered in the report (one per line):\n{topics_block}"
    
        messages = [
            SystemMessage(content=self.order_generator_prompt),
            HumanMessage(content=content)
        ]
        _assert_msgs(messages)
        response = self.llm.invoke(messages).content.strip()
        json_str = extract_json_from_code_block(response)
        try:
            data = json.loads(json_str)
            order_lst = data['order']
            return order_lst
        except:
            raise Exception(f"Invalid json format in generate_order: {json_str}")
    
    def generate_introduction(self, ordered_lst: List):
        print('Generating introduction...')
        messages = [
            SystemMessage(content = self.gen_introduction_prompt),
            HumanMessage(content= f"Here is the brief introduction of the dataset{self.introduction}\n\nAnd here is the ordered list of topics to be covered in the report:\n\n{str(ordered_lst)}"),
        ]
        _assert_msgs(messages)
        response = self.llm.invoke(messages).content.strip()
        m = re.search(r"<paragraph>\s*(.*?)\s*</paragraph>", response, flags=re.DOTALL | re.IGNORECASE)
        introduction = m.group(1) if m else None

        if not introduction:
            raise Exception(f"Invalid introduction format in generate_introduction: {response}")

        return introduction
    
    def narrative_compose(self, topic, image_path, insight):
        print("Generating narrative...")
        base64_image = encode_image(image_path)
        messages = [
            SystemMessage(content = self.gen_narrative_prompt),
            HumanMessage(content=[
                    {
                        "type": "text",
                        "text": f"Here is the topic of this section:\n\n{topic}\n\nAnd here is the insight list generated from the following chart\n\n{image_path}\n\nAnd here is the insight to be used in this section:\n\n{str(insight)}\n\n"
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
        _assert_msgs(messages)
        response = self.llm.invoke(messages).content.strip()
        m = re.search(r"<paragraph>\s*(.*?)\s*</paragraph>", response, flags=re.DOTALL | re.IGNORECASE)
        narrative = m.group(1) if m else None

        if not narrative:
            raise Exception(f"Could not extract narrative from response: {response}")
        
        return narrative

    def generate_transition(self, curr_topic, curr_narrative, next_topic, recent_transitions):
        print("Generate transition...")
        messages = [
            SystemMessage(content = self.transition_generator_prompt),
            HumanMessage(content= f"Here is the topic of current section:\n\n{curr_topic}\n\nAnd here is the narrative of current section:\n\n{curr_narrative}\n\nAnd here is the topic of next section:\n\n{next_topic}\n\nHere is the list of two recent transtitions(Pay attention to the cues used to avoid repitition): \n\n{recent_transitions}\n\n"),
        ]
        _assert_msgs(messages)
        response = self.llm.invoke(messages).content.strip()
        m = re.search(r"<paragraph>\s*(.*?)\s*</paragraph>", response, flags=re.DOTALL | re.IGNORECASE)
        transition = m.group(1) if m else None

        if not transition:
            raise Exception(f"Could not extract transition from response: {response}")

        return transition
    
    def rank_insight(self, curr_image_path):
        print("Ranking insights...")
        insight_lst = self.final_insight[curr_image_path]
        sorted_lst = sorted(insight_lst, key=lambda x: x['total_score'], reverse=True)
        insight_lst = [item['insight'] for item in sorted_lst[:3]]

        return insight_lst

    def conclusion_generation(self, introduction, topics_narratives):
        print("Generating conclusion...")
        body = "\n".join([f"Topic: {t}\nNarrative: {n}" for t, n, _ in topics_narratives])
        payload = f"Introduction Part: {introduction}\n\nTopics and Narratives:\n{body}"

        messages = [
            SystemMessage(content = self.conclusion_generator_prompt),
            HumanMessage(content= payload)
        ]
        _assert_msgs(messages)
        response = self.llm.invoke(messages).content.strip()
        m = re.search(r"<paragraph>\s*(.*?)\s*</paragraph>", response, flags=re.DOTALL | re.IGNORECASE)
        conclusion = m.group(1) if m else None

        if not conclusion:
            raise Exception(f"Could not extract conclusion from response: {response}")
        
        return conclusion
    
    def assemble(self, introduction, topics_narratives, conclusion):
        print("Assembling report...")
        lines = []
        fig_idx = 1
        # lines.append(f"*Generated on {date.today().isoformat()}*")
        lines.append("")
        lines.append("# Data Visualization Report")
        lines.append("")
        
        lines.append("## Introduction")
        lines.append(introduction.strip())
        lines.append("")

        for topic, narrative, img_path in topics_narratives:
            alt = topic.split(':')[0].strip()
            lines.append(f"## {fig_idx}. {alt}")
          
            lines.append(f"![Figure {fig_idx}: {alt}]({img_path})")
            # lines.append(f"Figure {fig_idx}: {alt}")

            fig_idx += 1
            lines.append("")

            lines.append(narrative.strip())
            lines.append("")
        
        # --- Conclusion
        lines.append("## Conclusion")
        lines.append(conclusion.strip())
        lines.append("")


        return "\n".join(lines)


    def process(self):
        self.initialize()
        print(f"Here is the topic list: {self.topic_lst}")
        ordered_lst = self.generate_order()
        print(f"Here is the ordered list: {ordered_lst}")
        introduction = self.generate_introduction(ordered_lst)
        print(f"Here is the introduction: {introduction}")
        topic_narratives = []
        
        norm = lambda s: " ".join(s.split()).strip().lower()
        norm_topics = [norm(x) for x in self.topic_lst]

        recent_transitions = []
        for i in range(len(ordered_lst)):
            print(f"Narrative the {i+1}-th topic: {ordered_lst[i]}")
            topic = ordered_lst[i]
            # index = self.topic_lst.index(topic)
            index = norm_topics.index(norm(topic))
            curr_image_path = self.verified_images_files[index]
            print(f"Here is the image path: {curr_image_path}")
            insight_lst = self.rank_insight(curr_image_path)
            narrative = self.narrative_compose(topic, curr_image_path, insight_lst)
            if i != len(ordered_lst) - 1:
                transition = self.generate_transition(topic, narrative, ordered_lst[i+1], recent_transitions[-2:])
                recent_transitions.append(transition)
                narrative = narrative + '\n' + transition
            print(f"Here is the narrative: {narrative}")

            topic_narratives.append((topic, narrative, curr_image_path))

        conclusion = self.conclusion_generation(introduction, topic_narratives)
        print(f"Here is the conclusion: {conclusion}")

        markdown_text = self.assemble(introduction, topic_narratives, conclusion)

        with open("report.md", "w", encoding="utf-8") as f:
            f.write(markdown_text)
        print(f"Markdown text has been saved to report.md")
        print("Transfer to pdf...")
        markdown_to_pdf(markdown_text, "output.pdf", os.getcwd())

        return markdown_text

                


        



        