import os

import yaml

_CURRENT_DIR_PATH = os.path.dirname(__file__)
_CONFIG_FILE_NAME = os.path.join(_CURRENT_DIR_PATH, ".config.yaml")


class Config:
    def __init__(self) -> None:
        self.config_kvs: dict = None

    def load(self) -> None:
        """各種変数を読み込む"""
        self.OPENAI_API_KEY = self._get_config("OPENAI_API_KEY")
        self.OPENAI_API_BASE = self._get_config("OPENAI_API_BASE")
        self.OPENAI_API_VERSION = self._get_config("OPENAI_API_VERSION")
        self.OPENAI_API_TYPE = self._get_config("OPENAI_API_TYPE")

    def _get_source(self):
        """
        .config.yaml が存在すれば、それを読み込む。それ以外の場合は、環境変数から設定を読み込む。

        .config.yaml は開発用途。本番では環境変数を使うこと。

        Returns:
            設定が格納されている辞書。
        """
        if os.path.exists(_CONFIG_FILE_NAME):
            # .config.yaml が存在するならば、そこから読み込む
            print(f"load from {_CONFIG_FILE_NAME}")
            with open(_CONFIG_FILE_NAME, "r") as f:
                config_kvs = yaml.safe_load(f)
        else:
            # そうでなければ、環境変数から読み込む。
            print("load from environment variables")
            config_kvs = os.environ
        return config_kvs

    def _get_config(self, key: str) -> str:
        """
        指定したキーに対応する設定値を取得する。設定が存在しない場合は、エラーメッセージを表示して終了する。

        Args:
            key (str): 取得したい設定のキー。

        Returns:
            str: 指定したキーに対応する設定値。

        Raises:
            KeyError: 指定したキーが設定に存在しない場合。
        """
        # self.config_kvs が None ならば、初期化する。
        if self.config_kvs is None:
            self.config_kvs = self._get_source()

        # key が config_kvs にあるかどうかを確認する。
        # なければメッセージを表示して終了する。
        if key not in self.config_kvs:
            msg = f"key: {key} が設定されていません"
            raise KeyError(msg)

        return self.config_kvs[key]


# singleton として利用する。
config = Config()
config.load()
