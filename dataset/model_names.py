model_names = [
    'robust_resnet',
    'resnet_salient_imagenet',

    'resnet50',
    'resnet101',

    'resnetv2_152x4_bitm',
    'resnetv2_152x4_bitm_in21k',

    'resnetv2_50x3_bitm',
    'resnetv2_50x3_bitm_in21k',

    'resnext50_32x4d',
    'resnext101_32x8d',
    'resnext101_64x4d',

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
    'convnext_large_mlp.clip_laion2b_augreg_ft_in1k',
    'convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384',

    'convnextv2_base.fcmae_ft_in22k_in1k_384',
    'convnextv2_large.fcmae_ft_in22k_in1k_384',
    'convnextv2_huge.fcmae_ft_in22k_in1k_384',

    'convnextv2_base.fcmae_ft_in1k',
    'convnextv2_large.fcmae_ft_in1k',
    'convnextv2_huge.fcmae_ft_in1k',

    'deit3_small_patch16_224',
    'deit3_small_patch16_224_in21ft1k',
    'deit3_large_patch16_384',
    'deit3_large_patch16_384_in21ft1k',

    'swin_base_patch4_window7_224_in22k',
    'swin_base_patch4_window7_224',

    'swin_large_patch4_window12_384_in22k',
    'swin_large_patch4_window12_384',

    'swinv2_large_window12to24_192to384_22kft1k',

    'vit_base_patch16_224.orig_in21k_ft_in1k',
    'vit_base_patch16_384.orig_in21k_ft_in1k',

    'vit_large_patch32_384.orig_in21k_ft_in1k',
    'vit_large_patch32_224.orig_in21k',

    'vit_large_patch16_224.augreg_in21k_ft_in1k',
    'vit_large_patch16_224.augreg_in21k',

    'vit_base_patch16_384.augreg_in1k',
    'vit_base_patch16_384.augreg_in21k_ft_in1k',
    'vit_base_patch8_224.augreg_in21k',

    'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k',

    'beit_base_patch16_224.in22k_ft_in22k_in1k',
    'beit_large_patch16_512.in22k_ft_in22k_in1k',

    'beit_base_patch16_224.in22k_ft_in22k',
    'beit_large_patch16_224.in22k_ft_in22k',

    'beitv2_large_patch16_224.in1k_ft_in22k_in1k',
    'beitv2_large_patch16_224.in1k_ft_in22k',
    
    'eva_giant_patch14_336.clip_ft_in1k',
    'eva_giant_patch14_560.m30m_ft_in22k_in1k',

    'volo_d5_512',
    'volo_d3_224',

    'ssl_resnext101_32x16d',
    'ssl_resnext50_32x4d',
    'ssl_resnet50',

    'swsl_resnext101_32x16d',
    'swsl_resnext50_32x4d',
    'swsl_resnet50',
]

model_names_table = {
    'robust_resnet':'Rob. ResNet50',
    'resnet_salient_imagenet':'Rob. ResNet50\\cite{Singla2022salient}',

    'resnet50':'ResNet50\\cite{HeZhaRen2015}',
    'resnet101':'ResNet101\\cite{HeZhaRen2015}',

    'resnetv2_152x4_bitm':'ResNetV2-152 BiT\\cite{KolesnikovEtAl2019}',
    'resnetv2_152x4_bitm_in21k':'ResNetV2-152 BiT\\cite{KolesnikovEtAl2019}',

    'resnetv2_50x3_bitm':'ResNetV2-50 BiT\\cite{KolesnikovEtAl2019}',
    'resnetv2_50x3_bitm_in21k':'ResNetV2-50 BiT\\cite{KolesnikovEtAl2019}',

    'resnext50_32x4d':'ResNeXt50 32x4d\\cite{xie2017aggregated}',
    'resnext101_32x8d':'ResNeXt101 32x8d\\cite{xie2017aggregated}',
    'resnext101_64x4d':'ResNeXt101 64x4d\\cite{xie2017aggregated}',

    'tf_efficientnet_b5.ra_in1k':'EfficientNet B5 RA\\cite{cubuk2020randaugment}',
    'tf_efficientnet_b5.ap_in1k':'EfficientNet B5 AP\\cite{xu2021adversarial}',
    'tf_efficientnet_b5.ns_jft_in1k':'EfficientNet B5 NS \\cite{xie2020selftraining}',

    'tf_efficientnet_b6.aa_in1k':'EfficientNet B6 AA\\cite{tan2019efficientnet}',
    'tf_efficientnet_b6.ap_in1k':'EfficientNet B6 AP\\cite{xu2021adversarial}',
    'tf_efficientnet_b6.ns_jft_in1k':'EfficientNet B6 NS \\cite{xie2020selftraining}',

    'tf_efficientnet_b7.ra_in1k':'EfficientNet B7 RA\\cite{cubuk2020randaugment}',
    'tf_efficientnet_b7.ap_in1k':'EfficientNet B7 AP\\cite{xu2021adversarial}',
    'tf_efficientnet_b7.ns_jft_in1k':'EfficientNet B7 NS \\cite{xie2020selftraining}',

    'tf_efficientnet_l2.ns_jft_in1k':'EfficientNet L2 NS \\cite{xie2020selftraining}',

    'tf_efficientnetv2_m.in1k':'EfficientNetV2-M\\cite{tan2021efficientnetv2}',
    'tf_efficientnetv2_m.in21k_ft_in1k':'EfficientNetV2-M\\cite{tan2021efficientnetv2}',
    'tf_efficientnetv2_m.in21k':'EfficientNetV2-M\\cite{tan2021efficientnetv2}',

    'tf_efficientnetv2_l.in1k':'EfficientNetV2-L\\cite{tan2021efficientnetv2}',
    'tf_efficientnetv2_l.in21k_ft_in1k':'EfficientNetV2-L\\cite{tan2021efficientnetv2}',
    'tf_efficientnetv2_l.in21k':'EfficientNetV2-L\\cite{tan2021efficientnetv2}',

    'convnext_base.fb_in1k':'ConvNeXt-B\\cite{liu2022convnet}',
    'convnext_base.fb_in22k_ft_in1k':'ConvNeXt-B\\cite{liu2022convnet}',
    'convnext_base.fb_in22k':'ConvNeXt-B\\cite{liu2022convnet}',

    'convnext_large.fb_in1k':'ConvNeXt-L\\cite{liu2022convnet}',
    'convnext_large.fb_in22k_ft_in1k':'ConvNeXt-L\\cite{liu2022convnet}',
    'convnext_large.fb_in22k':'ConvNeXt-L\\cite{liu2022convnet}',

    'convnext_xlarge.fb_in22k_ft_in1k':'ConvNeXt-XL\\cite{liu2022convnet}',
    'convnext_xlarge.fb_in22k':'ConvNeXt-XL\\cite{liu2022convnet}',

    'convnext_base.clip_laion2b_augreg_ft_in1k':'CNeXt-B CLIP\cite{radford2021learning} $\dagger$ ',
    'convnext_large_mlp.clip_laion2b_augreg_ft_in1k':'CNeXt-L CLIP\cite{radford2021learning} $\dagger$ 224',
    'convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384':'CNeXt-L CLIP\cite{radford2021learning} $\dagger$ 384',

    'convnextv2_base.fcmae_ft_in22k_in1k_384':'ConvNeXtV2-B\\cite{woo2023convnext}',
    'convnextv2_large.fcmae_ft_in22k_in1k_384':'ConvNeXtV2-L\\cite{woo2023convnext}',
    'convnextv2_huge.fcmae_ft_in22k_in1k_384':'ConvNeXtV2-H\\cite{woo2023convnext}',

    'convnextv2_base.fcmae_ft_in1k':'ConvNeXtV2-B\\cite{woo2023convnext}',
    'convnextv2_large.fcmae_ft_in1k':'ConvNeXtV2-L\\cite{woo2023convnext}',
    'convnextv2_huge.fcmae_ft_in1k':'ConvNeXtV2-H\\cite{woo2023convnext}',

    'deit3_small_patch16_224':'DeiT3-S\\textbackslash16 224\\cite{touvron2022deit3}',
    'deit3_small_patch16_224_in21ft1k':'DeiT3-S\\textbackslash16\\cite{touvron2022deit3}',
    'deit3_large_patch16_384':'DeiT3-L\\textbackslash16 384\\cite{touvron2022deit3}',
    'deit3_large_patch16_384_in21ft1k':'DeiT3-L\\textbackslash16\\cite{touvron2022deit3}',

    'swin_base_patch4_window7_224_in22k':'Swin-B 224\\cite{liu2021swin}',
    'swin_base_patch4_window7_224':'Swin-B 224\\cite{liu2021swin}',

    'swin_large_patch4_window12_384_in22k':'Swin-L 384\\cite{liu2021swin}',
    'swin_large_patch4_window12_384':'Swin-L 384\\cite{liu2021swin}',

    'swinv2_large_window12to24_192to384_22kft1k':'SwinV2-L\\cite{liu2022swinv2}',

    'vit_base_patch16_224.orig_in21k_ft_in1k':'ViT-B\\textbackslash16 224',
    'vit_base_patch16_384.orig_in21k_ft_in1k':'ViT-B\\textbackslash16 384\\cite{dosovitskiy2020image}',

    'vit_large_patch16_224.augreg_in21k_ft_in1k':'ViT-L\\textbackslash16 $\dagger$',
    'vit_large_patch16_224.augreg_in21k':'ViT-L\\textbackslash16 $\dagger$',

    'vit_base_patch16_384.augreg_in1k':'ViT-B\\textbackslash16 $\dagger$',
    'vit_base_patch16_384.augreg_in21k_ft_in1k':'ViT-B\\textbackslash16 $\dagger$',
    'vit_base_patch8_224.augreg_in21k':'ViT-B\\textbackslash8 $\dagger$',

    'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k':'ViT-L\\textbackslash14 CLIP\\cite{radford2021learning} 336',

    'beit_base_patch16_224.in22k_ft_in22k_in1k':'BEiT-B\\textbackslash16 224\\cite{bao2021beit}',
    'beit_large_patch16_512.in22k_ft_in22k_in1k':'BEiT-L\\textbackslash16\\cite{bao2021beit}',

    'beit_base_patch16_224.in22k_ft_in22k':'BEiT-B\\textbackslash16 224\\cite{bao2021beit}',
    'beit_large_patch16_224.in22k_ft_in22k':'BEiT-L\\textbackslash16 224\\cite{bao2021beit}',

    'beitv2_large_patch16_224.in1k_ft_in22k_in1k':'BEiTV2-L\\textbackslash16 224\\cite{peng2022beitv2}',
    'beitv2_large_patch16_224.in1k_ft_in22k':'BEiTV2-L\\textbackslash16 224\\cite{peng2022beitv2}',
    
    'eva_giant_patch14_336.clip_ft_in1k':'EVA-G\\textbackslash14 CLIP 336\\cite{EVA}',
    'eva_giant_patch14_560.m30m_ft_in22k_in1k':'EVA-G\\textbackslash14 CLIP 560\\cite{EVA}',
    
    'volo_d5_512':'VOLO-D5 512\\cite{yuan2022volo}',
    'volo_d3_224':'VOLO-D5 224\\cite{yuan2022volo}',

    'ssl_resnext101_32x16d':'ResNeXt101 SSL \\cite{yalniz2019billion}',
    'ssl_resnext50_32x4d':'ResNeXt50 SSL \\cite{yalniz2019billion}',
    'ssl_resnet50':'ResNet50 SSL \\cite{yalniz2019billion}',
    
    'swsl_resnext101_32x16d':'ResNeXt101 SSL \\cite{yalniz2019billion}',
    'swsl_resnext50_32x4d':'ResNeXt50 SSL \\cite{yalniz2019billion}',
    'swsl_resnet50':'ResNet50 SSL \\cite{yalniz2019billion}',  
}

