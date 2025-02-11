from Train import train, eval
from model_config import MODEL_CONFIG



def main(model_config):
    if model_config["state"] == "train":
        train(model_config)
        return None
    else:
        return eval(model_config)


if __name__ == "__main__":
    z_coes_array =main(MODEL_CONFIG)
    if z_coes_array is not None:
        print("eval finished!")
    else:
        print("train finished!")
