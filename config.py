import yaml

with open("config.yml", 'r') as file:
    config_data = yaml.load(file, Loader=yaml.FullLoader)

chat_model_config = config_data.get("chat_model_config", {})
embedding_model_config = config_data.get("embedding_model_config", {})
image_extraction_model = config_data.get("image_extraction_model", {})
