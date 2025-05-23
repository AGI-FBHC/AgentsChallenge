from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
from typing import Optional, Dict, List, Any, Union, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import threading
import queue
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

    def remove_urls_advanced(self,text: str) -> str:
        # 定义正则表达式以匹配网址
        url_pattern = r'(https?://\S+|www\.\S+)'
        # 分词处理，避免破坏其他部分
        tokens = text.split()
        cleaned_tokens = [token for token in tokens if not re.match(url_pattern, token)]
        return ' '.join(cleaned_tokens)

    def _thread_task(self, func, args, result_queue):
        """
        启动线程并将结果放入队列
        """
        result = func(*args)  # 调用实际函数
        result_queue.put(result)  # 将结果放入队列

    def generate_user_description(self, user_info: Dict[str, str], source: str) -> str:
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
        user_info_str = self.remove_urls_advanced(user_info_str)
        # 创建提示词
        prompt_filled = f"""The following is a user data of the {source} dataset after processing. Please write a paragraph in natural language describing the user. The description should cover their review activity (how many reviews they have written and their impact on the community, such as "useful," "funny," and "cool" votes), their elite status (mention the years they were considered Elite), and any notable aspects of their social circle (such as the number of friends they have). Make sure to convey the user's personality and activity on the platform, without including raw data, and keep it in a conversational tone,Your description should be fair and just,Keep your reply to around 50 words,Please describe in the second person:{user_info_str}"""
        # 调用模型生成响应
        retry_count = 0
        max_retries = 5
        success = False
        response = ""
        while retry_count < max_retries and not success:
            try:
                response = self.reasoning(prompt_filled)
                success = True  # 如果没有抛出异常，设置为成功
            except Exception as e:
                retry_count += 1
                print(f"Error occurred: {e}. Retry attempt {retry_count}/{max_retries}")
                if retry_count == max_retries:
                    print("Max retry attempts reached. The operation has failed.")

        return response

    def generate_item_description(self, item_info: Dict[str, str], source: str) -> str:
        """
        生成用户描述并返回响应。

        :param user_info: 包含用户信息的字典
        :param source: 数据源（例如 'yelp'）
        :return: 模型生成的自然语言描述
        """
        # 删除 'friends' 键

        # 将 user_info 转换为字符串
        item_info_str = str(item_info)
        item_info_str= self.remove_urls_advanced(item_info_str)
        # 创建提示词
        prompt_filled = f"""The following is a business data of the {source} dataset after processing. 
Please write a paragraph in natural language describing the business. 
If there are numbers in it, it is best to describe them.
The description should cover the name of the business, its location (address, city, and state), the type of business, its rating (stars), and the number of reviews. 
Mention the attributes such as whether it offers delivery, takeout, reservations, and whether it has a TV or parking. 
Also include any notable features like the ambiance, noise level, and what type of crowd it attracts (e.g., good for groups, casual, etc.). 
Make sure to describe the business in a way that reflects its character, customer experience, and atmosphere. 
Your description should be fair and just, and keep your reply to around 50 words:
{item_info_str}"""
        # 调用模型生成响应
        retry_count = 0
        max_retries = 5
        success = False
        response = ""
        while retry_count < max_retries and not success:
            try:
                response = self.reasoning(prompt_filled)
                success = True  # 如果没有抛出异常，设置为成功
            except Exception as e:
                retry_count += 1
                print(f"Error occurred: {e}. Retry attempt {retry_count}/{max_retries}")
                if retry_count == max_retries:
                    print("Max retry attempts reached. The operation has failed.")
        return response

    def generate_item_review_description(self, item_review_info: Dict[str, str], source: str) -> str:
        """
        生成用户描述并返回响应。

        :param user_info: 包含用户信息的字典
        :param source: 数据源（例如 'yelp'）
        :return: 模型生成的自然语言描述
        """
        # 删除 'friends' 键
        random_item_reviews = random.sample(item_review_info, min(len(item_review_info), 10))
        # random_user_reviews = random.sample(user_review_info, min(len(user_review_info), 10))
        # 将 user_info 转换为字符串
        item_review_info = str(random_item_reviews)
        item_review_info = self.remove_urls_advanced(item_review_info)
        # 创建提示词
        prompt_filled = f"""The following is a collection of user reviews for the business  from the {source} dataset. 
        Please write a comprehensive summary of the reviews in natural language. 
        Your description should:
        1. Summarize the general sentiment (positive or negative) expressed across the reviews.
        2. Highlight recurring themes or issues (e.g., poor service, food quality, atmosphere).
        3. Mention any specific strengths or weaknesses pointed out by multiple reviewers (e.g., friendly staff, slow service, great food).
        4. If applicable, include the most common ratings or complaints (e.g., average rating, issues with cleanliness or pricing).
        5. Avoid repeating the same information and keep the summary balanced and fair, reflecting both positive and negative feedback.
        Please ensure the review summary is clear, concise, and informative, with a word limit of around 300 words:
        {item_review_info}"""
        # 调用模型生成响应
        # 调用模型生成响应
        retry_count = 0
        max_retries = 5
        success = False
        response = ""
        while retry_count < max_retries and not success:
            try:
                response = self.reasoning(prompt_filled)
                success = True  # 如果没有抛出异常，设置为成功
            except Exception as e:
                retry_count += 1
                print(f"Error occurred: {e}. Retry attempt {retry_count}/{max_retries}")
                if retry_count == max_retries:
                    print("Max retry attempts reached. The operation has failed.")

        return response

    def generate_user_review_description(self, user_review_info: Dict[str, str], source: str) -> str:
        """
        生成用户描述并返回响应。

        :param user_info: 包含用户信息的字典
        :param source: 数据源（例如 'yelp'）
        :return: 模型生成的自然语言描述
        """
        # 删除 'friends' 键
        # random_item_reviews = random.sample(item_reviews, min(len(item_reviews), 5))
        random_user_reviews = random.sample(user_review_info, min(len(user_review_info), 10))
        # 将 user_info 转换为字符串
        user_review_info = str(random_user_reviews)
        user_review_info= self.remove_urls_advanced(user_review_info)
        # 创建提示词
        prompt_filled = f"""The following is a collection of historical reviews written by a user from the {source} dataset. 
        Please analyze the user's reviewing style and summarize their characteristics. 
        Focus on the following aspects:
        1. The user's tone and language (e.g., formal, casual, critical, humorous).
        2. Common themes or topics the user frequently comments on (e.g., food quality, service, atmosphere).
        3. The level of detail in the reviews (e.g., whether they provide specific examples or remain general).
        4. The user's sentiment (e.g., balanced, overly positive, overly negative).
        5. Any patterns in the user's ratings (e.g., tends to give moderate scores, extremes, or consistent ratings).
        6. How the user evaluates positives and negatives in the experience.
        Based on this analysis, provide a concise summary of the user's reviewing style in natural language, highlighting their typical approach to writing reviews. Limit the summary to around 50 words,Please describe in the second person:
        {user_review_info}"""
        # 调用模型生成响应
        retry_count = 0
        max_retries = 5
        success = False
        response = ""
        while retry_count < max_retries and not success:
            try:
                response = self.reasoning(prompt_filled)
                success = True  # 如果没有抛出异常，设置为成功
            except Exception as e:
                retry_count += 1
                print(f"Error occurred: {e}. Retry attempt {retry_count}/{max_retries}")
                if retry_count == max_retries:
                    print("Max retry attempts reached. The operation has failed.")

        return response

    def extract_review(self, task_description, max_retries=2):
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

            item_reviews_text = ""
            user_reviews_text = ""

            # 获取用户来源信息
            source = user_info.get('source', '')

            result_queue = queue.Queue()

            # 创建线程列表
            threads = [
                threading.Thread(target=self._thread_task,
                                 args=(self.generate_user_description, (user_info, source), result_queue)),
                threading.Thread(target=self._thread_task,
                                 args=(self.generate_item_description, (item_info, source), result_queue)),
                threading.Thread(target=self._thread_task,
                                 args=(self.generate_item_review_description, (item_reviews, source), result_queue)),
                threading.Thread(target=self._thread_task,
                                 args=(self.generate_user_review_description, (user_reviews, source), result_queue))
            ]

            # 启动所有线程
            for thread in threads:
                thread.start()

            # 等待所有线程完成并获取结果
            for thread in threads:
                thread.join()

            # 从队列中获取结果
            after_user_info = result_queue.get()
            item_description = result_queue.get()
            item_reviews_text = result_queue.get()
            user_reviews_text = result_queue.get()

            task_description = f'''
        You are a user on {source}, a platform for crowd-sourced business reviews. Your task is to write a review for the business described below. The first section provides information about you, including your past activities on the platform:

        {after_user_info},{user_reviews_text}.
        
        Business Overview:
        {item_description}
        
        Summary of Previous Reviews:
        {item_reviews_text}
        
        Your Past Reviews:
        Here are some of your past reviews for other businesses. You can refer to them to simulate your review style: {random_user_reviews}
        
        Instructions:
        Rating: Based on your profile and past review style, provide a rating for this business.
        Many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for poor experiences that fail to meet basic standards.
        Be cautious when giving a 5-star rating, ensuring the experience truly exceeds expectations.
        Avoid giving 1-star ratings unless there are significant issues in key areas that negatively impact the business. Consider providing constructive feedback for businesses that fall short of expectations.
        Review Aspects: After analyzing the business details and your past experiences, focus on the aspects that make this business stand out—positively or negatively.
        Highlight specific elements that stood out to you, such as product quality, service, or customer experience.
        Ensure your review is an objective evaluation based on the quality of the experience.
        Review Requirements:
        Star Rating: Must be one of: 1.0, 2.0, 3.0, 4.0, or 5.0. The rating should reflect an honest and balanced assessment of the business.
        Review Length: The review should be 3-4 sentences, about 180 tokens, with a focus on your personal experience and emotional response.
        Consistency: Maintain consistency with your historical review style and rating patterns.
        Comments Should Include:
        Topic Relevance: Your review must reflect the overall evaluation of the business and include the business name. Specify the subject of your review.
        Sentiment: Your review should reflect the overall sentiment (positive or negative) based on the product quality and other feedback from users.
        Emotional Tone: Your initial emotional response should be expressed at the beginning of your review. Ensure the tone matches your overall evaluation of the business.
        Format your response exactly as follows:
        stars: [your rating]
        review: [your review]
        '''
            stars, review_text = self.extract_review(task_description)

            if len(review_text) > 512:
                review_text = review_text[:512]

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
