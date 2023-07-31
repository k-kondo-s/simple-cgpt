"""
観点は
* 普通にメッセージが返ってくるか
* ちゃんと trim するか
* 長いメッセージ history でも正常に動くか。
"""

from langchain.schema import AIMessage, HumanMessage

from agent import GPT_4, GPT_35_TURBO, Agent


def test_trim_messages():
    """テストメソッド、trim_messagesが指定したトークン数に基づいて正しくメッセージをトリムすることを検証する。

    このテストケースでは、4つのメッセージ（2つのHumanMessageと2つのAIMessage）が与えられ、
    トリム処理後の結果が正しいかどうかをチェックします。結果としては、最初のHumanMessageが削除され、
    残りのメッセージがそのままの順序で返されるべきです。

    Args:
        なし

    Raises:
        AssertionError: 期待される結果と異なる場合に発生します。
    """
    messages = [
        HumanMessage(content=str(10**100 * 1)),
        AIMessage(content=str(10**100 * 2)),
        HumanMessage(content=str(10**100 * 3)),
        AIMessage(content=str(10**100 * 4)),
    ]
    agent = Agent()

    # ここでは最初のメッセージがトリムされる。
    result_messages_1 = agent.trim_messages(messages, max_tokens=160, max_tokens_response=10, llm=agent.gpt_4)
    assert result_messages_1[0].content == str(10**100 * 2)
    assert result_messages_1[1].content == str(10**100 * 3)
    assert result_messages_1[2].content == str(10**100 * 4)

    # ここでは何もトリムされない。
    result_messages_2 = agent.trim_messages(messages, max_tokens=160, max_tokens_response=10, llm=agent.gpt_4)
    assert result_messages_2[0].content == str(10**100 * 2)
    assert result_messages_2[1].content == str(10**100 * 3)
    assert result_messages_2[2].content == str(10**100 * 4)


def test_agent_call():
    """エージェントが与えられたメッセージを処理し、指定されたモデルを使用して正しい応答を返すかテストします。

    このテストメソッドは、異なるモデル（GPT_4およびGPT_35_TURBO）を使用してエージェントが
    同じメッセージを正しく処理するかどうかを検証します。結果としては、どちらのモデルも同じ結果を返すべきです。

    Args:
        なし

    Raises:
        AssertionError: 期待される結果と異なる場合に発生します。

    Note:
        以下は正常系のテストのみ。
        また LLM なので、 assert が常に通るとは原理的には限らない。
        厳密にはそうであるが、しかしながら、以下のテストケースが通らないとなると
        何かしらモデルに著しい変化があったとみなしてよかろう。
    """
    agent = Agent()
    messages = [
        HumanMessage(content=str(10**100 * 1)),
        AIMessage(content=str(10**100 * 2)),
        HumanMessage(content=str(10**100 * 3)),
        AIMessage(content=str(10**100 * 4)),
        HumanMessage(content=str(10**100 * 5)),
    ]
    result_gpt4 = agent(messages=messages, model=GPT_4)
    assert result_gpt4 == str(10**100 * 6)
    result_gpt35_turbo = agent(messages=messages, model=GPT_35_TURBO)
    assert result_gpt35_turbo == str(10**100 * 6)
