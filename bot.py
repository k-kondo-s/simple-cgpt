# %%
import os
from typing import List, Tuple

from multi_agent import MultiAgent
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from config import config

# 環境変数から Slack の API キーを取得

SLACK_BOT_TOKEN = config.SLACK_BOT_TOKEN
SLACK_APP_TOKEN = config.SLACK_APP_TOKEN


# Slack のアプリを初期化
app = App(token=SLACK_BOT_TOKEN)
qa = MultiAgent()


def build_output_message(result: dict, event, is_error=False) -> str:
    """返信するメッセージを作成する"""
    # エラーがあれば、エラーメッセージを返す
    if is_error:
        return result

    # source_documents にあるそれぞれのファイル名を取得する。
    # 例えば、'root/data/2023年/企画書.pdf' というファイル名があれば、企画書.pdf という文字列にする。
    _source_documents = [r.metadata["source"] for r in result["source_documents"] if "source" in r.metadata]
    source_documents = set([os.path.basename(s) for s in _source_documents])

    # message を構築する
    result_text = result.get("answer", "") or result.get("result", "")  # chain によって key の名前が違うめんどくささがある
    message = f'<@{event["user"]}>\n'
    message += f"{result_text}\n\n"
    if len(source_documents) > 0:
        message += "ヒントになりそうな文書:\n"
        for s in source_documents:
            message += f"- {s}\n"
    return message


def extract_chat_history(event, app) -> List[Tuple]:
    """chat_history を抽出する"""
    # thread 情報を取得する
    thread_ts = event.get("thread_ts", None)
    thread_info = app.client.conversations_replies(channel=event["channel"], ts=thread_ts)

    # bot の user_id を取得する
    bot_id = app.client.auth_test().data["bot_id"]

    # chat_history を作る。
    # まず bot が返した message を thread_info から取得する。このとき、その index も取得する。
    bot_messages = []
    for index, m in enumerate(thread_info.data["messages"]):
        if "bot_id" in m and m["bot_id"] == bot_id:
            bot_messages.append((index, m))

    # bot_message の直前の message を user_message とする。
    user_messages = []
    for index, m in bot_messages:
        if index == 0:
            continue
        user_message = thread_info.data["messages"][index - 1]
        user_messages.append((index - 1, user_message))

    # (user_message, bot_message) のタプルを作る。
    chat_history = [(u[1]["text"], b[1]["text"]) for u, b in zip(user_messages, bot_messages)]

    # 無尽蔵に履歴が増えて token があふれることを防ぐために、 chat_history の最後のいくつかだけ残す
    if len(chat_history) > config.MESSAGE_HISTORY_COUNT:
        chat_history = chat_history[-config.MESSAGE_HISTORY_COUNT :]

    return chat_history


@app.event("app_mention")
def respond_to_mention(event, say):
    """chatbotにメンションが付けられたときのハンドラ"""
    print(event)

    # スレッドの中身を取得する。
    chat_history = []
    if event.get("thread_ts", None):
        chat_history = extract_chat_history(event, app)

    # 質問に対する回答を取得する
    is_error = False
    try:
        result = qa.run(event["text"], chat_history)
    except Exception as e:
        is_error = True
        result = f"エラーがおきました :しゅん: \n```{e.args}\n```"
    print(result)

    # output メッセージを構築する
    output = build_output_message(result, event, is_error=is_error)

    # メッセージを送信。スレッドに返信する。
    thread_ts = event.get("thread_ts", None) or event["ts"]
    say(text=output, thread_ts=thread_ts, reply_broadcast=True)


@app.event("message")
def handle_message_events(body, logger):
    logger.info(body)


# Slack のアプリを起動
SocketModeHandler(app, SLACK_APP_TOKEN).start()
