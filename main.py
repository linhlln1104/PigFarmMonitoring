if __name__ == '__main__':
    from models.yolov9_model import train_model
    from config.CFG import ModelConfig
    from utils.create_yaml import create_yaml_file

    create_yaml_file()
    train_model(ModelConfig)
