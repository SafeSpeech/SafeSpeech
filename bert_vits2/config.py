"""
@Desc: 全局配置文件读取
"""

import argparse
import yaml
from typing import Dict, List
import os
import shutil
import sys


class Resample_config:
    def __init__(self, in_dir: str, out_dir: str, sampling_rate: int = 44100):
        self.sampling_rate: int = sampling_rate  # 目标采样率
        self.in_dir: str = in_dir  # 待处理音频目录路径
        self.out_dir: str = out_dir  # 重采样输出路径

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["in_dir"] = os.path.join(dataset_path, data["in_dir"])
        data["out_dir"] = os.path.join(dataset_path, data["out_dir"])

        return cls(**data)


class Preprocess_text_config:

    def __init__(
        self,
        transcription_path: str,
        cleaned_path: str,
        train_path: str,
        val_path: str,
        config_path: str,
        val_per_lang: int = 5,
        max_val_total: int = 10000,
        clean: bool = True,
    ):
        self.transcription_path: str = (
            transcription_path
        )
        self.cleaned_path: str = (
            cleaned_path
        )
        self.train_path: str = (
            train_path
        )
        self.val_path: str = (
            val_path
        )
        self.config_path: str = config_path
        self.val_per_lang: int = val_per_lang
        self.max_val_total: int = (
            max_val_total
        )
        self.clean: bool = clean

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):

        data["transcription_path"] = os.path.join(
            dataset_path, data["transcription_path"]
        )
        if data["cleaned_path"] == "" or data["cleaned_path"] is None:
            data["cleaned_path"] = None
        else:
            data["cleaned_path"] = os.path.join(dataset_path, data["cleaned_path"])
        data["train_path"] = os.path.join(dataset_path, data["train_path"])
        data["val_path"] = os.path.join(dataset_path, data["val_path"])
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Bert_gen_config:


    def __init__(
        self,
        config_path: str,
        num_processes: int = 2,
        device: str = "cuda",
        use_multi_device: bool = False,
    ):
        self.config_path = config_path
        self.num_processes = num_processes
        self.device = device
        self.use_multi_device = use_multi_device

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Emo_gen_config:


    def __init__(
        self,
        config_path: str,
        num_processes: int = 2,
        device: str = "cuda",
        use_multi_device: bool = False,
    ):
        self.config_path = config_path
        self.num_processes = num_processes
        self.device = device
        self.use_multi_device = use_multi_device

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Train_ms_config:


    def __init__(
        self,
        config_path: str,
        env: Dict[str, any],
        base: Dict[str, any],
        model: str,
        num_workers: int,
        spec_cache: bool,
        keep_ckpts: int,
    ):
        self.env = env
        self.base = base
        self.model = (
            model
        )
        self.config_path = config_path
        self.num_workers = num_workers
        self.spec_cache = spec_cache
        self.keep_ckpts = keep_ckpts

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # data["model"] = os.path.join(dataset_path, data["model"])
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Webui_config:

    def __init__(
        self,
        device: str,
        model: str,
        config_path: str,
        language_identification_library: str,
        port: int = 7860,
        share: bool = False,
        debug: bool = False,
    ):
        self.device: str = device
        self.model: str = model
        self.config_path: str = config_path
        self.port: int = port
        self.share: bool = share
        self.debug: bool = debug
        self.language_identification_library: str = (
            language_identification_library
        )

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["config_path"] = os.path.join(dataset_path, data["config_path"])
        data["model"] = os.path.join(dataset_path, data["model"])
        return cls(**data)


class Server_config:
    def __init__(
        self, models: List[Dict[str, any]], port: int = 5000, device: str = "cuda"
    ):
        self.models: List[Dict[str, any]] = models
        self.port: int = port
        self.device: str = device

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class Translate_config:

    def __init__(self, app_key: str, secret_key: str):
        self.app_key = app_key
        self.secret_key = secret_key

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class Config:
    def __init__(self, config_path: str):
        if not os.path.isfile(config_path) and os.path.isfile("default_config.yml"):
            shutil.copy(src="default_config.yml", dst=config_path)
            print(
                f"已根据默认配置文件default_config.yml生成配置文件{config_path}。请按该配置文件的说明进行配置后重新运行。"
            )
            print("如无特殊需求，请勿修改default_config.yml或备份该文件。")
            sys.exit(0)
        with open(file=config_path, mode="r", encoding="utf-8") as file:
            yaml_config: Dict[str, any] = yaml.safe_load(file.read())
            dataset_path: str = yaml_config["dataset_path"]
            openi_token: str = yaml_config["openi_token"]
            self.dataset_path: str = dataset_path
            self.mirror: str = yaml_config["mirror"]
            self.openi_token: str = openi_token
            self.resample_config: Resample_config = Resample_config.from_dict(
                dataset_path, yaml_config["resample"]
            )
            self.preprocess_text_config: Preprocess_text_config = (
                Preprocess_text_config.from_dict(
                    dataset_path, yaml_config["preprocess_text"]
                )
            )
            self.bert_gen_config: Bert_gen_config = Bert_gen_config.from_dict(
                dataset_path, yaml_config["bert_gen"]
            )
            self.emo_gen_config: Emo_gen_config = Emo_gen_config.from_dict(
                dataset_path, yaml_config["emo_gen"]
            )
            self.train_ms_config: Train_ms_config = Train_ms_config.from_dict(
                dataset_path, yaml_config["train_ms"]
            )
            self.webui_config: Webui_config = Webui_config.from_dict(
                dataset_path, yaml_config["webui"]
            )
            self.server_config: Server_config = Server_config.from_dict(
                yaml_config["server"]
            )
            self.translate_config: Translate_config = Translate_config.from_dict(
                yaml_config["translate"]
            )


parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yml_config", type=str, default="bert_vits2/config.yml")
args, _ = parser.parse_known_args()
config = Config(args.yml_config)
