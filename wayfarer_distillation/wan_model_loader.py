from wan.modules.model import WanModel


def load_wan_model(model_name=None):
    if not model_name:
        model_name = "Wan2.1-T2V-14B"
        model_name = "Wan2.1-T2V-1.3B"
    model = WanModel.from_pretrained(f"Wan-AI/{model_name}", repo_type='namespace/repo_name')
    # model = WanModel.from_pretrained(f"wan_models/{model_name}/")
    return model

if __name__ == "__main__":
    model = load_wan_model()
    
    print(model)