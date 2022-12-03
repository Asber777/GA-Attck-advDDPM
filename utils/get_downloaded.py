import timm
def get_architecture(model_name="InceptionV3"):

    ############################
    # Attacking models in Tab. 1 
    ############################

    ################################
    # Eight Naturally trained models
    ################################

    # Three Transformers
    if model_name == "vit_small_patch16_224":
        model = timm.create_model('vit_small_patch16_224', pretrained=True)
        model.input_size = 224

    elif model_name == "vit_base_patch16_224":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.input_size = 224
    elif model_name == "swin_base_patch4_window7_224":
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        model.input_size = 224
    
    # Three ResNets
    elif model_name == "swsl_resnext101_32x8d":
        model = timm.create_model('swsl_resnext101_32x8d', pretrained=True)
        model.input_size = 224
    
    elif model_name == "ssl_resnext50_32x4d":
        model = timm.create_model('ssl_resnext50_32x4d', pretrained=True)
        model.input_size = 224
    
    elif model_name == "swsl_resnet50":
        model = timm.create_model('swsl_resnet50', pretrained=True)
        model.input_size = 224
    
    # Inception
    elif model_name == "InceptionV3":
        model = timm.create_model('tf_inception_v3', pretrained=True)
        model.input_size = 299
    
    elif model_name == "InceptionResnetV2":
        model = timm.create_model('inception_resnet_v2', pretrained=True)
        model.input_size = 299
    
    ###########################################
    # Two Ensemble Adversarially trained models
    ###########################################

    elif model_name == "adv_inception_v3":
        model = timm.create_model('adv_inception_v3', pretrained=True)
        model.input_size = 299
    
    elif model_name == "ens_adv_inception_resnet_v2":
        model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
        model.input_size = 299

MODEL_NAME_DICT = {
    0: "vit_small_patch16_224",
    1: "vit_base_patch16_224",
    2: "swin_base_patch4_window7_224",
    3: "swsl_resnext101_32x8d",
    4: "ssl_resnext50_32x4d",
    5: "swsl_resnet50",
    6: "InceptionV3",
    7: "InceptionResnetV2",
    8: "adv_inception_v3",
    9: "ens_adv_inception_resnet_v2",
}
source_id_list = [i for i in range(0, 10)]
for idx in source_id_list:
    print("downloading {}".format(MODEL_NAME_DICT[idx]))
    get_architecture(model_name=MODEL_NAME_DICT[idx])
    print("{} is done.".format(MODEL_NAME_DICT[idx]))