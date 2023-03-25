model_names = [
    # # ##
    'robust_resnet',
    'resnet_salient_imagenet',
    'spurious_projection_robust_resnet',

    # ##old-school
    'resnet50',
    'resnet101',

    # ##Bit
    'resnetv2_152x4_bitm',
    'resnetv2_152x4_bitm_in21k',

    'resnetv2_50x3_bitm',
    'resnetv2_50x3_bitm_in21k',

    # # resnext 1k
    'resnext50_32x4d',
    'resnext101_32x8d',
    'resnext101_64x4d',

    # ##efficient nets
    'tf_efficientnet_b5.ra_in1k',
    'tf_efficientnet_b5.ap_in1k',
    'tf_efficientnet_b5.ns_jft_in1k',

    'tf_efficientnet_b6.aa_in1k',
    'tf_efficientnet_b6.ap_in1k',
    'tf_efficientnet_b6.ns_jft_in1k',

    'tf_efficientnet_b7.ra_in1k',
    'tf_efficientnet_b7.ap_in1k',
    'tf_efficientnet_b7.ns_jft_in1k',

    'tf_efficientnet_l2.ns_jft_in1k',

    'tf_efficientnetv2_m.in1k',
    'tf_efficientnetv2_m.in21k_ft_in1k',
    'tf_efficientnetv2_m.in21k',

    'tf_efficientnetv2_l.in1k',
    'tf_efficientnetv2_l.in21k_ft_in1k',
    'tf_efficientnetv2_l.in21k',

    'convnext_base.fb_in1k',
    'convnext_base.fb_in22k_ft_in1k',
    'convnext_base.fb_in22k',

    'convnext_large.fb_in1k',
    'convnext_large.fb_in22k_ft_in1k',
    'convnext_large.fb_in22k',

    'convnext_xlarge.fb_in22k_ft_in1k',
    'convnext_xlarge.fb_in22k',

    'convnext_base.clip_laion2b_augreg_ft_in1k',
    #'convnext_base.clip_laiona_augreg_ft_in1k_384',
    'convnext_large_mlp.clip_laion2b_augreg_ft_in1k',
    'convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384',

    #convnext-v2
    'convnextv2_base.fcmae_ft_in22k_in1k_384',
    'convnextv2_large.fcmae_ft_in22k_in1k_384',
    'convnextv2_huge.fcmae_ft_in22k_in1k_384',

    'convnextv2_base.fcmae_ft_in1k',
    'convnextv2_large.fcmae_ft_in1k',
    'convnextv2_huge.fcmae_ft_in1k',

    # 'convnextv2_base.fcmae',
    # 'convnextv2_large.fcmae',
    # 'convnextv2_huge.fcmae',

    #deit3
    'deit3_small_patch16_224',
    'deit3_small_patch16_224_in21ft1k',
    'deit3_large_patch16_384',
    'deit3_large_patch16_384_in21ft1k',

    'swin_base_patch4_window7_224_in22k',
    'swin_base_patch4_window7_224',

    'swin_large_patch4_window12_384_in22k',
    'swin_large_patch4_window12_384',

    'swinv2_large_window12to24_192to384_22kft1k',

    ##VIT
    'vit_base_patch16_224.orig_in21k_ft_in1k',
    'vit_base_patch16_384.orig_in21k_ft_in1k',

    'vit_large_patch32_384.orig_in21k_ft_in1k',
    'vit_large_patch32_224.orig_in21k',


    'vit_large_patch16_224.augreg_in21k_ft_in1k',
    'vit_large_patch16_224.augreg_in21k',

    'vit_base_patch16_384.augreg_in1k',
    'vit_base_patch16_384.augreg_in21k_ft_in1k',
    'vit_base_patch8_224.augreg_in21k',

    #
    'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k',
    #'vit_large_patch14_clip_336.openai_ft_in12k_in1k',

    ##pretrained on21k
    'beit_base_patch16_224.in22k_ft_in22k_in1k',
    'beit_large_patch16_512.in22k_ft_in22k_in1k',

    'beit_base_patch16_224.in22k_ft_in22k',
    'beit_large_patch16_224.in22k_ft_in22k',

    'beitv2_large_patch16_224.in1k_ft_in22k_in1k',
    'beitv2_large_patch16_224.in1k_ft_in22k',
    
    'eva_giant_patch14_336.clip_ft_in1k',
    'eva_giant_patch14_560.m30m_ft_in22k_in1k',
    
    ##only in-1k
    'volo_d5_512',
    'volo_d3_224',

    # # # SSL pre-trained on YFCC100M
    'ssl_resnext101_32x16d',
    'ssl_resnext50_32x4d',
    'ssl_resnet50',
    #
    # ##Billion-scale SSL
    'swsl_resnext101_32x16d',
    'swsl_resnext50_32x4d',
    'swsl_resnet50',

]
