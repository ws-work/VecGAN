"""
Configurations file defining the options for the model. The options specified 
here denotes the number of channels in the generator and the label organization.
The generator is constructed using these options.
"""

#data 
input_dim = 3
tags = [{
            "name": "Bangs",
            "tag_irrelevant_conditions_dim": 2,
            "attributes": [
                {
                    "name": "with",
                    "filename": "datasets/Bangs_with.txt"
                },
                {
                    "name": "without",
                    "filename": "datasets/Bangs_without.txt"
                }
            ]
        },
        {
            "name": "Eyeglasses",
            "tag_irrelevant_conditions_dim": 2,
            "attributes": [
                {
                    "name": "with",
                    "filename": "datasets/Eyeglasses_with.txt"
                },
                {
                    "name": "without",
                    "filename": "datasets/Eyeglasses_without.txt"
                }
            ]
        },
        {
            "name": "HairColor",
            "tag_irrelevant_conditions_dim": 2,
            "attributes": [
                {
                    "name": "black",
                    "filename": "datasets/HairColor_black.txt"
                },
                {
                    "name": "blond",
                    "filename": "datasets/HairColor_blond.txt"
                },
                {
                    "name": "brown",
                    "filename": "datasets/HairColor_brown.txt"
                }
            ]
        },
        {
            "name": "Male",
            "tag_irrelevant_conditions_dim": 2,
            "attributes": [
                {
                    "name": "with",
                    "filename": "datasets/Male_with.txt"
                },
                {
                    "name": "without",
                    "filename": "datasets/Male_without.txt"
                }
            ]
        },
        {
            "name": "Smiling",
            "tag_irrelevant_conditions_dim": 2,
            "attributes": [
                {
                    "name": "with",
                    "filename": "datasets/Smiling_with.txt"
                },
                {   
                    "name": "without",
                    "filename": "datasets/Smiling_without.txt"
                }
            ]
        },

    ]

# gen
channel_size = 3
img_dim = 256
latent_dim = 2048
encoder_channels = [32, 64, 128, 256, 512, 512, 512, 1024, 2048]
decoder_channels = [2048, 1024, 512, 512, 512, 256, 128, 64, 32]

style_dim = 256


# dis
discriminators_channels = [64, 128, 256, 512, 1024, 2048]

# trian
batch_size = 8
new_size = 128
crop_image_height = 128 
crop_image_width = 128 
total_iterations = 200000

beta1 = 0
beta2 = 0.99

lr_dis = 0.0001
lr_gen_mappers = 0.000001
lr_gen_others = 0.0001

noise_dim = 32

# Label Organization
## The tags and attributes are labeled with this index order.
# tags = [
#     {
#         "name": "Bangs", # Tag 0
#         "attributes": ["with", "without"] # Attributes 0, 1
#     },
#     {
#         "name": "Eyeglasses", # Tag 1
#         "attributes": ["with", "without"] # Attributes 0, 1
#     },
#     {
#         "name": "Hair_Color", # Tag 2
#         "attributes": ["black", "brown", "blond"] # Attributes 0, 1, 2
#     },
#     {
#         "name": "Young", # Tag 3
#         "attributes": ["with", "without"] # Attributes 0, 1
#     },
#     {
#         "name": "Male", # Tag 4
#         "attributes": ["with", "without"] # Attributes 0, 1
#     },
#     {
#         "name": "Smiling", # Tag 5
#         "attributes": ["with", "without"] # Attributes 0, 1
#     },
# ]