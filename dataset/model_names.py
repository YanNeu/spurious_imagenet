model_names = [
    # ##
    'robust_resnet',
    'resnet_salient_imagenet',
    'spurious_projection_robust_resnet',
    #
    # ##old-school
    'resnet50',
    'resnet101',
    #
    # ##Bit
    'resnetv2_152x4_bitm',
    'resnetv2_152x4_bitm_in21k',

    'resnetv2_50x3_bitm',
    'resnetv2_50x3_bitm_in21k',
    #
    # # resnext 1k
    'resnext50_32x4d',
    'resnext101_32x8d',
    'resnext101_64x4d',
    #
    # ##efficient nets
    'tf_efficientnet_b5',
    'tf_efficientnet_b5_ap',
    #'tf_efficientnet_b5_ns',

    'tf_efficientnet_b6',
    'tf_efficientnet_b6_ap',
    #'tf_efficientnet_b6_ns',

    'tf_efficientnet_b7',
    'tf_efficientnet_b7_ap',
    #'tf_efficientnet_b7_ns',

    #'tf_efficientnet_l2_ns',

    'tf_efficientnetv2_m',
    'tf_efficientnetv2_m_in21ft1k',
    'tf_efficientnetv2_m_in21k',

    'tf_efficientnetv2_l',
    'tf_efficientnetv2_l_in21ft1k',
    'tf_efficientnetv2_l_in21k',
    #
    'convnext_base',
    'convnext_base_in22ft1k',

    'convnext_large',
    'convnext_large_in22ft1k',
    'convnext_large_in22k',

    'convnext_xlarge_384_in22ft1k',
    'convnext_xlarge_in22k',

    'deit3_small_patch16_224',
    'deit3_small_patch16_224_in21ft1k',
    'deit3_large_patch16_384',
    'deit3_large_patch16_384_in21ft1k',

    'swin_base_patch4_window7_224_in22k',
    'swin_base_patch4_window7_224',

    'swin_large_patch4_window12_384_in22k',
    'swin_large_patch4_window12_384',

    # 'swinv2_large_window12to24_192to384_22kft1k',

    ##all vits are 21k pretrained
    'vit_base_patch16_224',
    'vit_base_patch16_384',
    'vit_base_patch16_224_in21k',

    'vit_large_patch16_384',
    'vit_large_patch16_224',
    'vit_large_patch16_224_in21k',

    ##pretrained on21k
    'beit_base_patch16_224',
    'beit_large_patch16_512',

    'beit_base_patch16_224_in22k',
    'beit_large_patch16_224_in22k',

    ##only in-1k
    'volo_d5_512',
    'volo_d3_224',

    # # SSL pre-trained on YFCC100M
    # 'ssl_resnext101_32x16d',
    # 'ssl_resnext50_32x4d',
    # 'ssl_resnet50',
    #
    # ##Billion-scale SSL
    # 'swsl_resnext101_32x16d',
    # 'swsl_resnext50_32x4d',
    # 'swsl_resnet50',

]
