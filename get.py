import pandas as pd
import logging
import os
import random
import json
from typing import Optional, Dict, List, Any,Union
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("websocietysimulator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class ReviewParser:
    """
    一个用于解析输入文本并提取 'stars' 和 'review' 的类。
    """

    def __init__(self):
        """
        初始化 ReviewParser。
        """
        pass

    def parse(self, input_text: str) -> Dict[str, Any]:
        """
        解析输入文本，提取 'stars' 和 'review'。

        :param input_text: 语言模型生成的输出字符串。
        :return: 包含 'stars' 和 'review' 的字典。如果解析失败，则返回默认值。
        """
        try:
            # 使用正则表达式提取 stars
            stars_pattern = re.compile(r'stars:\s*([1-5](?:\.\d)?)', re.IGNORECASE)
            stars_match = stars_pattern.search(input_text)
            if stars_match:
                stars = float(stars_match.group(1))
                logging.info(f"提取到的 stars: {stars}")
            else:
                logging.error("未能从输入中提取到 'stars'。使用默认值 3.0。")
                stars = 3.0  # 默认评分

            # 使用正则表达式提取 review
            review_pattern = re.compile(r'review:\s*(.*)', re.IGNORECASE | re.DOTALL)
            review_match = review_pattern.search(input_text)
            if review_match:
                review_text = review_match.group(1).strip()
                # 移除多余的空白字符，包括换行符
                review_text = re.sub(r'\s+', ' ', review_text)
                # 限制评论长度为 512 字符
                if len(review_text) > 512:
                    review_text = review_text[:512]
                    logging.warning("评论文本超过 512 字符，已被截断。")
                logging.info(f"提取到的 review: {review_text}")
            else:
                logging.error("未能从输入中提取到 'review'。使用默认空字符串。")
                review_text = ""

            return {
                "stars": stars,
                "review": review_text
            }

        except Exception as e:
            logging.error(f"解析过程中发生异常: {e}")
            return {
                "stars": 3.0,  # 默认评分
                "review": "评价生成失败。"
            }


class InteractionTool:
    def __init__(self, data_dir: str):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
        """
        logger.info(f"Initializing InteractionTool with data directory: {data_dir}")
        self.data_dir = data_dir
        # Convert DataFrames to dictionaries for O(1) lookup
        logger.info(f"Loading item data from {os.path.join(data_dir, 'item.json')}")
        self.item_data = {item['item_id']: item for item in self._load_data('item.json')}
        logger.info(f"Loading user data from {os.path.join(data_dir, 'user.json')}")
        self.user_data = {user['user_id']: user for user in self._load_data('user.json')}

        # Create review indices
        logger.info(f"Loading review data from {os.path.join(data_dir, 'review.json')}")
        reviews = self._load_data('review.json')
        self.review_data = {review['review_id']: review for review in reviews}
        self.item_reviews = {}
        self.user_reviews = {}

        # Build review indices
        logger.info("Building review indices")
        for review in reviews:
            # Index by item_id
            self.item_reviews.setdefault(review['item_id'], []).append(review)
            # Index by user_id
            self.user_reviews.setdefault(review['user_id'], []).append(review)

    def _load_data(self, filename: str) -> List[Dict]:
        """Load data as a list of dictionaries."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Fetch user data based on user_id."""
        return self.user_data.get(user_id)

    def get_item(self, item_id: str = None) -> Optional[Dict]:
        """Fetch item data based on item_id."""
        return self.item_data.get(item_id) if item_id else None

    def get_reviews(
            self,
            item_id: Optional[str] = None,
            user_id: Optional[str] = None,
            review_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch reviews filtered by various parameters."""
        if review_id:
            return [self.review_data[review_id]] if review_id in self.review_data else []

        if item_id:
            return self.item_reviews.get(item_id, [])
        elif user_id:
            return self.user_reviews.get(user_id, [])

        return []


class LLMBase:
    def __init__(self, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize LLM base class

        Args:
            model: Model name, defaults to qwen2.5-72b-instruct
        """
        self.model = model

    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0,
                 max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call LLM to get response

        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            temperature: Sampling temperature, defaults to 0.0
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1

        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        raise NotImplementedError("Subclasses need to implement this method")


class InfinigenceLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "qwen2.5-72b-instruct"):
        """
        Initialize Infinigence LLM

        Args:
            api_key: API key for authentication
            model: Model name, defaults to qwen2.5-72b-instruct
        """
        super().__init__(model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://cloud.infini-ai.com/maas/v1"
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=4, max=60),  # 等待时间从4秒开始，指数增长，最长60秒
        stop=stop_after_attempt(5)  # 最多重试5次
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0,
                 max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call Infinigence AI API to get response with rate limit handling

        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            temperature: Sampling temperature, defaults to 0.0
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1

        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_strs,
                n=n
            )

            if n == 1:
                return response.choices[0].message.content
            else:
                return [choice.message.content for choice in response.choices]
        except Exception as e:
            if "429" in str(e):
                logger.warning("Rate limit exceeded")
            else:
                logger.error(f"Other LLM Error: {e}")
            raise e







class LLMClient:
    def __init__(self, api_key=None, model="qwen2.5-72b-instruct", system_message="You are a helpful assistant."):
        """
        初始化 LLM 客户端实例。
        :param api_key: API 密钥，可以直接传入，也可以从环境变量中读取。
        :param model: 模型名称，默认为 'qwen2.5-72b-instruct'。
        :param system_message: 系统消息，用于设置对话背景。
        """
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Please provide it or set it in the environment as 'API_KEY'.")
        self.model = model
        self.system_message = system_message
        self.client = InfinigenceLLM(api_key=self.api_key, model=model)

    def generate_response(self, content, temperature=0.7, max_tokens=150):
        """
        调用模型生成响应。
        :param content: 用户输入的内容。
        :param temperature: 控制生成内容的随机性。
        :param max_tokens: 生成内容的最大 token 数量。
        :return: 模型响应。
        """
        # 构建消息格式
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": content}
        ]

        try:
            response = self.client(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            return f"Error: {e}"

class InteractionProcessor:
    def __init__(self, data_dir):
        """
        Initialize the InteractionProcessor with the data directory.

        :param data_dir: Path to the data directory.
        """
        self.interaction_tool = InteractionTool(data_dir=data_dir)
        self.prompt ='''
        You are a real human user on {source} dataset, a platform for crowd-sourced business reviews. Here is your  profile and review history: 
        {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}
            Here are some other items you have reviewed{your_review}
            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            3. Consider how other users might engage with your review in terms of:
            - Useful: How informative and helpful is your review?
            - Funny: Does your review have any humorous or entertaining elements?
            - Cool: Is your review particularly insightful or praiseworthy?

            Requirements:
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
            - If the business fails significantly in key areas, consider giving a 1-star rating
            - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Format your response exactly as follows:
            stars: [your rating]
            useful: [count]
            funny: [count] 
            cool: [count]
            review: [your review]
        '''

    def process(self, user_id, item_id):
        """
        Process the interaction by fetching user info, item info, and reviews.

        :param user_id: The user ID to fetch information for.
        :param item_id: The item ID to fetch information for.
        :return: A dictionary containing user info, item info, and reviews.
        """
        try:
            # Fetch user and item information
            user_info = self.interaction_tool.user_data.get(user_id)
            item_info = self.interaction_tool.item_data.get(item_id)
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            random_item_reviews = random.sample(item_reviews, min(len(item_reviews), 5))
            random_user_reviews = random.sample(user_reviews, min(len(user_reviews), 5))
            source = user_info.get('source','unknown')
            prompt1 = self.prompt.format(source = source,user = str(user_id),business =str(item_info),review_similar =str(random_item_reviews),your_review = str(random_user_reviews))
            # Fetch reviews for the specific item and user
            llm_client = LLMClient()  # 或从环境变量中获取
            # 用户输入内容

            # 调用模型生成响应
            response = llm_client.generate_response(content=prompt1)
            parser = ReviewParser()
            results = parser.parse(response)
            # Return all gathered information
            return results
        except Exception as e:
            print(f"Error during processing: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    # 如果需要，可直接设置环境变量 API_KEY
    # os.environ["API_KEY"] = "your_api_key_here"

    # 实例化类


    # 初始化 InteractionProcessor
    data_dir = "/home/zhaorh/code/AgentSociety/home/zhaorh/code/AgentSociety/dataout"
    processor = InteractionProcessor(data_dir=data_dir)

    # 输入的 user_id 和 item_id
    user_id = "wAo7casDFsbUR4O8Vb3u8A"
    item_id = "cVoyA9wrdF5E8OroBahxCg"

    # 调用处理方法
    results = processor.process(user_id=user_id, item_id=item_id)
    print(f"Stars: {results['stars']}")
    print(f"Review: {results['review']}")

