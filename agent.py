from typing import Dict, List, Optional, Union
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import os
logger = logging.getLogger("websocietysimulator")


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


# 使用示例
if __name__ == "__main__":
    # 如果需要，可直接设置环境变量 API_KEY
    # os.environ["API_KEY"] = "your_api_key_here"

    # 实例化类
    llm_client = LLMClient()  # 或从环境变量中获取

    # 用户输入内容
    user_content = "What is an LLM?"

    # 调用模型生成响应
    response = llm_client.generate_response(content=user_content)
    print("Model Response:", response)

