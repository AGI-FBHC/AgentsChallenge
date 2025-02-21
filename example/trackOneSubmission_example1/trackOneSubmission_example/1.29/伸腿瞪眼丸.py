from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
from typing import Optional, Dict, List, Any, Union, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import random
import re
import json
import logging

logging.basicConfig(level=logging.INFO)

# 设置日志文件路径
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    filename='application.log',  # 指定日志文件名
    filemode='a'  # 设置文件打开模式，'a'表示追加
)


class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""

    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)

    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan


class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )

        return reasoning_result


class MySimulationAgent(SimulationAgent):
    """Participant's implementation of SimulationAgent."""

    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)

    def remove_urls_advanced(self, text: str) -> str:
        # 定义正则表达式以匹配网址
        url_pattern = r'(https?://\S+|www\.\S+)'
        # 分词处理，避免破坏其他部分
        tokens = text.split()
        cleaned_tokens = [token for token in tokens if not re.match(url_pattern, token)]
        return ' '.join(cleaned_tokens)
    def generate_combined_description(self, user_info: Dict[str, str], item_info: Dict[str, str],
                                      item_review_info: Dict[str, str], user_review_info: Dict[str, str],
                                      source: str) -> str:
        """
        生成用户描述并返回响应。

        :param user_info: 包含用户信息的字典
        :param source: 数据源（例如 'yelp'）
        :return: 模型生成的自然语言描述
        """
        # 删除 'friends' 键
        if 'friends' in user_info:
            del user_info['friends']

        # 将 user_info 转换为字符串
        user_info_str = str(user_info)
        item_info_str = str(item_info)
        random_item_reviews = random.sample(item_review_info, min(len(item_review_info), 10))
        item_review_info = str(random_item_reviews)
        random_user_reviews = random.sample(user_review_info, min(len(user_review_info), 10))
        user_review_info = str(random_user_reviews)
        user_info_str =self.remove_urls_advanced(user_info_str)
        item_info_str =self.remove_urls_advanced(item_info_str)
        item_review_info = self.remove_urls_advanced(item_review_info)
        user_review_info =self.remove_urls_advanced(user_review_info)
        # 创建提示词
        prompt_filled = f"""
        You are provided with processed data from the {source} dataset:

        1. User Information:
           {user_info_str}

        2. User Review History:
           {user_review_info}

        3. Business Information:
           {item_info_str}

        4. Collection of Business Reviews:
           {item_review_info}

        Your response must be in **two paragraphs** with an **objective tone**:

        - **Paragraph One (~120 tokens):**
          - Write in the **second person** to portray the user，refer to User Review History and User Information.
          - Include details on activity level (such as how many reviews you have contributed), reviewing style or preferences, sentiment (positive, negative, or neutral), and any notable rating patterns.
          - Maintain a factual, objective tone without being overly personal or casual.
          - Additionally, please clarify how the user tends to rate their experience with star ratings and provide an explanation.

        - **Paragraph Two (~380 tokens total):**
          - First (~80 tokens): Provide a concise description of the business. Mention its name, location (address, city, state), type of business, star rating, approximate number of reviews (preferably in words), and key attributes.You need to pay attention to mention the star rating of this merchant.
          - Next (~300 tokens): Summarize the overall sentiment and recurring themes in the reviews, including positive and negative feedback. Highlight major strengths and weaknesses observed, and indicate any consistent praise or complaints. Keep the tone balanced and avoid repetitive details.

        **Important**: Do not include any headings or labels like “Task 1” or “Task 2” in your final response. Simply produce two paragraphs in the specified order, each fulfilling the respective requirements.
        """

        prompt_filled2 = f"""
                You are provided with processed data from a virtual platform:

                1. User Information:
                   {user_info_str}

                2. User Review History:
                   {user_review_info}

                3. Business Information:
                   {item_info_str}

                4. Collection of Business Reviews:
                   {item_review_info}

                Your response must be in **two paragraphs** with an **objective tone**:

                - **Paragraph One (~120 tokens):**
                  - Write in the **second person** to portray the user，refer to User Review History and User Information.
                  - Include details on activity level (such as how many reviews you have contributed), reviewing style or preferences, sentiment (positive, negative, or neutral), and any notable rating patterns.
                  - Maintain a factual, objective tone without being overly personal or casual.
                  - Additionally, please clarify how the user tends to rate their experience with star ratings and provide an explanation.

                - **Paragraph Two (~380 tokens total):**
                  - First (~80 tokens): Provide a concise description of the business. Mention its name, location (address, city, state), type of business, star rating, approximate number of reviews (preferably in words), and key attributes.You need to pay attention to mention the star rating of this merchant.
                  - Next (~300 tokens): Summarize the overall sentiment and recurring themes in the reviews, including positive and negative feedback. Highlight major strengths and weaknesses observed, and indicate any consistent praise or complaints. Keep the tone balanced and avoid repetitive details.

                **Important**: Do not include any headings or labels like “Task 1” or “Task 2” in your final response. Simply produce two paragraphs in the specified order, each fulfilling the respective requirements.
                """
        # 调用模型生成响应
        retry_count = 0
        max_retries = 5
        success = False
        response = ""
        while retry_count < max_retries and not success:
            try:
                if source.lower() in ['yelp', 'amazon', 'goodreads']:
                    response = self.reasoning(prompt_filled)  # 如果 source 是指定的值，使用 prompt_filled
                    logging('是在数据集里面')
                else:
                    response = self.reasoning(prompt_filled2)  # 否则，使用 prompt_filled2
                    logging('不是在数据集里面')
                success = True  # 如果没有抛出异常，设置为成功
            except Exception as e:
                retry_count += 1
                print(f"Error occurred: {e}. Retry attempt {retry_count}/{max_retries}")
                if retry_count == max_retries:
                    print("Max retry attempts reached. The operation has failed.")

        return response

    def extract_review(self, task_description, max_retries=3):
        retries = 0
        while retries < max_retries:
            retry_count = 0
            success = False

            # 尝试获取结果，带有重试机制
            while retry_count < max_retries and not success:
                try:
                    result = self.reasoning(task_description)  # 调用reasoning方法获取结果
                    success = True  # 如果没有抛出异常，设置为成功
                except Exception as e:
                    retry_count += 1
                    logging.error(f"发生错误: {e}. 重试次数 {retry_count}/{max_retries}")
                    if retry_count == max_retries:
                        logging.error("已达到最大重试次数，操作失败。")

            # 如果reasoning成功，尝试提取stars和review
            if success:
                logging.info(f"提取的结果: {result}")
                try:
                    # 提取包含stars和review的行
                    stars_line = next(line for line in result.split('\n') if 'stars:' in line)
                    review_line = next(line for line in result.split('\n') if 'review:' in line)
                    review_text = review_line.replace('review:', '').strip()  # 去掉前缀并去除多余的空格

                    # 提取stars的数值并强制转换为整数
                    stars_value = float(stars_line.split(':')[1].strip())  # 将stars值转换为整数
                    logging.info(f"提取的评分：{stars_value}")

                    return stars_value, review_text  # 返回整数评分和评论
                except StopIteration:
                    retries += 1
                    logging.warning(f"错误: 无法提取评分或评论。重试次数 {retries}/{max_retries}...")
                    if retries >= max_retries:
                        logging.error('已达到最大重试次数，返回None。')
                        return None, None  # 如果提取失败，返回None
            else:
                # 如果reasoning失败，则开始下次重试
                retries += 1
                logging.warning(f"reasoning失败，重试次数 {retries}/{max_retries}...")
                if retries >= max_retries:
                    logging.error('已达到最大重试次数，返回None。')
                    return None, None

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            user_id = self.task.get('user_id')
            item_id = self.task.get('item_id')

            # 获取相关评论
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            user_info = self.interaction_tool.get_user(user_id=user_id)
            item_info = self.interaction_tool.get_item(item_id=item_id)
            # 随机选择一些评论以供参考
            # random_item_reviews = random.sample(item_reviews, min(len(item_reviews), 5))
            random_user_reviews = random.sample(user_reviews, min(len(user_reviews), 3))

            # 获取用户来源信息
            source = user_info.get('source', '')
            # after_user_info = self.generate_user_description(user_info, source)
            # item_description = self.generate_item_description(item_info, source)
            # = self.generate_item_review_description(item_reviews, source)
            # user_reviews_text = self.generate_user_review_description(user_reviews, source)
            com = self.generate_combined_description(user_info, item_info, item_reviews, user_reviews, source)
            task_description1 = f'''
        You are a real human user on {source} dataset, a platform for crowd-sourced business reviews.You are very good at thinking.Your task is to write a review for a business described below. The first paragraph is a description of you and your past behavior,along with some feedback you have received in the past. while the second paragraph describes the product you need to evaluate: 
        {com},
            ##Here are some of the user's past reviews of other businesses, you can use these to simulate the user's style.{random_user_reviews}##
            You need to write a review for this business,
            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            Requirements:
            - Attention, Attention.You must be objective when scoring, and not just give high scores blindly.
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - Attention: Please be cautious when giving 5 stars, unless the experience exceeds expectations.
            - Avoid giving 1 star reviews.
            - Review text should be 3-4 sentences, focusing on your personal experience and emotional response,Review are about 180 tokens.
            - Maintain consistency with your historical review style and rating patterns.
            - Please provide an objective evaluation of the star rating, do not add extra stars out of friendliness.
            Comments must meet the following requirements:
            1. Topic Relevance:"reflects the overall user evaluation of the merchant. You must specify the subject of your comment. The review must include the name of the business.You must specify the subject of your comment."
            2. Sentiment Attitude:"Generate a review that reflects positive or negative sentiment based on product quality and ensures alignment with overall user feedback."
            3. Emotional Tone:"express your initial emotion or feeling about the experience in front of the sentence. The emotion should be placed right after the business name and before describing the specifics of your experience. Be sure to include your overall evaluation of the business."

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
        '''
            task_description2 = f'''
                    You are a real human user, a platform for crowd-sourced business reviews.You are very good at thinking.Your task is to write a review for a business described below. The first paragraph is a description of you and your past behavior,along with some feedback you have received in the past. while the second paragraph describes the product you need to evaluate: 
                    {com},
                        ##Here are some of the user's past reviews of other businesses, you can use these to simulate the user's style.{random_user_reviews}##
                        You need to write a review for this business,
                        Please analyze the following aspects carefully:
                        1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations.
                        2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
                        Requirements:
                        - Attention, Attention.You must be objective when scoring, and not just give high scores blindly.
                        - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
                        - Attention: Please be cautious when giving 5 stars, unless the experience exceeds expectations.
                        - Review text should be 3-4 sentences, focusing on your personal experience and emotional response,Review are about 200 tokens.
                        - Maintain consistency with your historical review style and rating patterns.
                        - Please provide an objective evaluation of the star rating, do not add extra stars out of friendliness.
                        Comments must meet the following requirements:
                        1. Topic Relevance:"reflects the overall user evaluation of the merchant. You must specify the subject of your comment. The review must include the name of the business.You must specify the subject of your comment."
                        2. Sentiment Attitude:"Generate a review that reflects positive or negative sentiment based on product quality and ensures alignment with overall user feedback."
                        3. Emotional Tone:"express your initial emotion or feeling about the experience in front of the sentence. The emotion should be placed right after the business name and before describing the specifics of your experience. Be sure to include your overall evaluation of the business."

                        Format your response exactly as follows:
                        stars: [your rating]
                        review: [your review]
                    '''
            if source.lower() in ['yelp', 'amazon', 'goodreads']:
                stars, review_text = self.extract_review(task_description1)
            else:
                stars, review_text = self.extract_review(task_description2)



            if len(review_text) > 512:
                review_text = review_text[:512]

            """logging.info("one: %s", after_user_info)
            logging.info("two: %s", item_description)
            logging.info("two: %s",item_info)
            logging.info("three: %s", user_reviews_text)
            logging.info("four: %s", item_reviews_text)"""
            logging.info("one:%s", com)
            if stars and review_text:
                return {
                    "stars": stars,
                    "review": review_text
                }
            else:
                return {
                    "stars": 3,
                    "review": "The product meets basic expectations and serves its intended purpose. While it doesn't stand out in terms of exceptional quality or features, it provides decent value for the price. Overall, it is a reliable option for everyday use. I would recommend it to those looking for a practical solution without premium expectations."
                }

        except Exception as e:
            print(f"Error in workflow: {e}")
            return {
                "stars": 3,
                "review": "The product meets basic expectations and serves its intended purpose. While it doesn't stand out in terms of exceptional quality or features, it provides decent value for the price. Overall, it is a reliable option for everyday use. I would recommend it to those looking for a practical solution without premium expectations."
            }
