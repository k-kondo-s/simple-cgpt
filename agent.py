# %%
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage

from config import config

# Azure OpenAI のデプロイメント名
GPT_35_TURBO = "azure-gpt-35-turbo-16k"
GPT_4 = "azure-gpt-4-32k"

# それぞれのデプロイメントの最大トークン数
# https://learn.microsoft.com/ja-jp/azure/cognitive-services/openai/concepts/models#model-capabilities
MAX_TOKENS_GPT_35_TURBO = 16_384
MAX_TOKENS_GPT_4 = 32_768

# GPT が返すトークン数の最大値
# Slack の一回の投稿で送れる最大文字数は 4000 文字なので、それに合わせて、
# 少し余裕を持たせて 3950 文字としている。
MAX_TOKENS_RESPONSE_GPT_35_TURBO = 3950
MAX_TOKENS_RESPONSE_GPT_4 = 3950


class Agent:
    def __init__(
        self,
    ) -> None:
        self.gpt_35_turbo = AzureChatOpenAI(
            deployment_name=GPT_35_TURBO,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=config.OPENAI_API_BASE,
            openai_api_version=config.OPENAI_API_VERSION,
            openai_api_type=config.OPENAI_API_TYPE,
            max_tokens=MAX_TOKENS_RESPONSE_GPT_35_TURBO,
        )

        self.gpt_4 = AzureChatOpenAI(
            deployment_name=GPT_4,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=config.OPENAI_API_BASE,
            openai_api_version=config.OPENAI_API_VERSION,
            openai_api_type=config.OPENAI_API_TYPE,
            max_tokens=MAX_TOKENS_RESPONSE_GPT_4,
        )

    def get_llm_props_from_model(self, model: str) -> tuple[AzureChatOpenAI, int, int]:
        if model == GPT_35_TURBO:
            return self.gpt_35_turbo, MAX_TOKENS_GPT_35_TURBO, MAX_TOKENS_RESPONSE_GPT_35_TURBO
        elif model == GPT_4:
            return self.gpt_4, MAX_TOKENS_GPT_4, MAX_TOKENS_RESPONSE_GPT_4
        else:
            raise ValueError(f"Invalid model name: {model}")

    def trim_messages(self, messages: list, max_tokens: int, max_tokens_response: int, llm: AzureChatOpenAI) -> list:
        """トークン数が最大値を超えているメッセージをトリムする"""
        # 現時点の token 数が max_tokens からレスポンスの最大トークン数を引いた値より大きい場合は、
        # 最も古いメッセージを削除する。
        current_tokens = llm.get_num_tokens_from_messages(messages)
        while current_tokens > max_tokens - max_tokens_response:
            messages.pop(0)
            current_tokens = llm.get_num_tokens_from_messages(messages)
        return messages

    def __call__(self, messages: list, model: str) -> str:
        llm, max_tokens, max_tokens_response = self.get_llm_props_from_model(model)
        messages = self.trim_messages(messages, max_tokens, max_tokens_response, llm)
        result = llm(messages=messages)
        return result.content


messages = [
    HumanMessage(content="質問1: あなたの生年月日を累乗したら？"),
    AIMessage(content="質問1の答え"),
    HumanMessage(content="あなたのGPTのversionは？"),
    AIMessage(content="OpenAI"),
    HumanMessage(content="続きを"),
]
agent = Agent()
agent(messages=messages, model=GPT_4)

# %%
