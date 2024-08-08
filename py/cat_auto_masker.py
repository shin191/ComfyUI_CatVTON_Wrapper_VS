

from .func import *
NODE_NAME = 'CatVTON_AutoMasker'
class MaskGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "cloth_type": (["upper", "lower", "overall","inner","outer"], {"default": "upper"}),
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("mask","densepose",)
    FUNCTION = "automasker"
    CATEGORY = '😺dzNodes/CatVTON AutoMasker'
    def automasker(self, person_image, cloth_type):
        catvton_path = os.path.join(folder_paths.models_dir, "CatVTON")
        automasker = AutoMasker(
            densepose_ckpt=os.path.join(catvton_path, "DensePose"),
            schp_ckpt=os.path.join(catvton_path, "SCHP"),
            device='cuda', 
        )
        mask, densepose =automasker( person_image,cloth_type)
        log(f"{NODE_NAME} Processed.", message_type='finish')
        return (mask, densepose,)


NODE_CLASS_MAPPINGS = {
    "CatAutoMasker": MaskGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CatAutoMasker": "CatVTON AutoMasker"
}    