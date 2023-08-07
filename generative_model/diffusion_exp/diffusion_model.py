# '''
# File Name: diffusion_model
# Create File Time: 2023/5/29 9:33
# File Create By Author: Yang Guanqun
# Email: yangguanqun01@corp.netease.com
# Corp: Fuxi Tech, Netease
# '''
#
#
# import torch
#
# from diffusers import AutoencoderKL
# ckpt_path = r"E:\github\lora-scripts\sd-models\model.ckpt"
#
# def load_checkpoint_with_text_encoder_conversion(ckpt_path):
#     TEXT_ENCODER_KEY_REPLACEMENTS = [
#         ('cond_stage_model.transformer.embeddings.', 'cond_stage_model.transformer.text_model.embeddings.'),
#         ('cond_stage_model.transformer.encoder.', 'cond_stage_model.transformer.text_model.encoder.'),
#         ('cond_stage_model.transformer.final_layer_norm.', 'cond_stage_model.transformer.text_model.final_layer_norm.')
#     ]
#
#     checkpoint = torch.load(ckpt_path, map_location="cpu")
#     if "state_dict" in checkpoint:
#         state_dict = checkpoint['state_dict']
#     # print(state_dict.keys())
#
#     key_reps = []
#     for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
#         for key in state_dict.keys():
#             if key.startswith(rep_from):
#                 new_key = rep_to + key[len(rep_from):]
#                 key_reps.append((key, new_key))
#
#     for key, new_key in key_reps:
#         state_dict[new_key] = state_dict[key]
#         del state_dict[key]
#     # checkpoint包含各类成员变量，state_dict主要是网络参数
#     return checkpoint, state_dict
#
# def shave_segments(path, n_shave_prefix_segments=1):
#   """
#   Removes segments. Positive values shave the first segments, negative shave the last segments.
#   """
#   if n_shave_prefix_segments >= 0:
#     return ".".join(path.split(".")[n_shave_prefix_segments:])
#   else:
#     return ".".join(path.split(".")[:n_shave_prefix_segments])
#
# def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
#   """
#   Updates paths inside attentions to the new naming scheme (local renaming)
#   """
#   mapping = []
#   for old_item in old_list:
#     new_item = old_item
#
#     new_item = new_item.replace("norm.weight", "group_norm.weight")
#     new_item = new_item.replace("norm.bias", "group_norm.bias")
#
#     new_item = new_item.replace("q.weight", "query.weight")
#     new_item = new_item.replace("q.bias", "query.bias")
#
#     new_item = new_item.replace("k.weight", "key.weight")
#     new_item = new_item.replace("k.bias", "key.bias")
#
#     new_item = new_item.replace("v.weight", "value.weight")
#     new_item = new_item.replace("v.bias", "value.bias")
#
#     new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
#     new_item = new_item.replace("proj_out.bias", "proj_attn.bias")
#
#     new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)
#
#     mapping.append({"old": old_item, "new": new_item})
#
#   return mapping
#
# def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
#   """
#   Updates paths inside resnets to the new naming scheme (local renaming)
#   """
#   mapping = []
#   for old_item in old_list:
#     new_item = old_item
#
#     new_item = new_item.replace("nin_shortcut", "conv_shortcut")
#     new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)
#
#     mapping.append({"old": old_item, "new": new_item})
#
#   return mapping
#
# def assign_to_checkpoint(
#     paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
# ):
#   """
#   This does the final conversion step: take locally converted weights and apply a global renaming
#   to them. It splits attention layers, and takes into account additional replacements
#   that may arise.
#
#   Assigns the weights to the new checkpoint.
#   """
#   assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."
#
#   # Splits the attention layers into three variables.
#   if attention_paths_to_split is not None:
#     for path, path_map in attention_paths_to_split.items():
#       old_tensor = old_checkpoint[path]
#       channels = old_tensor.shape[0] // 3
#
#       target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)
#
#       num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3
#
#       old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
#       query, key, value = old_tensor.split(channels // num_heads, dim=1)
#
#       checkpoint[path_map["query"]] = query.reshape(target_shape)
#       checkpoint[path_map["key"]] = key.reshape(target_shape)
#       checkpoint[path_map["value"]] = value.reshape(target_shape)
#
#   for path in paths:
#     new_path = path["new"]
#
#     # These have already been assigned
#     if attention_paths_to_split is not None and new_path in attention_paths_to_split:
#       continue
#
#     # Global renaming happens here
#     new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
#     new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
#     new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")
#
#     if additional_replacements is not None:
#       for replacement in additional_replacements:
#         new_path = new_path.replace(replacement["old"], replacement["new"])
#
#     # proj_attn.weight has to be converted from conv 1D to linear
#     if "proj_attn.weight" in new_path:
#       checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
#     else:
#       checkpoint[new_path] = old_checkpoint[path["old"]]
#
# def conv_attn_to_linear(checkpoint):
#   keys = list(checkpoint.keys())
#   attn_keys = ["query.weight", "key.weight", "value.weight"]
#   for key in keys:
#     if ".".join(key.split(".")[-2:]) in attn_keys:
#       if checkpoint[key].ndim > 2:
#         checkpoint[key] = checkpoint[key][:, :, 0, 0]
#     elif "proj_attn.weight" in key:
#       if checkpoint[key].ndim > 2:
#         checkpoint[key] = checkpoint[key][:, :, 0]
#
# def convert_ldm_vae_checkpoint(state_dict, config):
#     vae_state_dict = {}
#     vae_key = "first_stage_model."
#     keys = list(state_dict.keys())
#     # first_state_model表示VAE model
#     for key in keys:
#         if key.startswith(vae_key):
#             vae_state_dict[key.replace(vae_key, "")] = state_dict.get(key)
#
#     new_checkpoint = {}
#
#     new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
#     new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
#     new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
#     new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
#     new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
#     new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]
#
#     new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
#     new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
#     new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
#     new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
#     new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
#     new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]
#
#     new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
#     new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
#     new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
#     new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]
#
#
#     num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
#     # encoder的blocks
#     down_blocks = {
#         layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
#     }
#
#     num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
#     # decoder的blocks
#     up_blocks = {
#         layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
#     }
#
#     for i in range(num_down_blocks):
#         resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]
#
#         if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
#             new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
#                 f"encoder.down.{i}.downsample.conv.weight"
#             )
#             new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
#                 f"encoder.down.{i}.downsample.conv.bias"
#             )
#
#         paths = renew_vae_resnet_paths(resnets)
#         meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
#         assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
#
#     mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
#     num_mid_res_blocks = 2
#     for i in range(1, num_mid_res_blocks + 1):
#         resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]
#
#         paths = renew_vae_resnet_paths(resnets)
#         meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
#         assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
#
#     mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
#     paths = renew_vae_attention_paths(mid_attentions)
#     meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
#     assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
#     conv_attn_to_linear(new_checkpoint)
#
#     for i in range(num_up_blocks):
#         block_id = num_up_blocks - 1 - i
#         resnets = [
#             key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
#         ]
#
#         if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
#             new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
#                 f"decoder.up.{block_id}.upsample.conv.weight"
#             ]
#             new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
#                 f"decoder.up.{block_id}.upsample.conv.bias"
#             ]
#
#         paths = renew_vae_resnet_paths(resnets)
#         meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
#         assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
#
#     mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
#     num_mid_res_blocks = 2
#     for i in range(1, num_mid_res_blocks + 1):
#         resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]
#
#         paths = renew_vae_resnet_paths(resnets)
#         meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
#         assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
#
#     mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
#     paths = renew_vae_attention_paths(mid_attentions)
#     meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
#     assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
#     conv_attn_to_linear(new_checkpoint)
#     return new_checkpoint
#
# def create_vae_diffusers_config():
#     VAE_PARAMS_Z_CHANNELS = 4
#     VAE_PARAMS_RESOLUTION = 256
#     VAE_PARAMS_IN_CH = 3
#     VAE_PARAMS_OUT_CH = 3
#     VAE_PARAMS_CH = 128
#     VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
#     VAE_PARAMS_NUM_RES_BLOCKS = 2
#
#     block_out_channels = [VAE_PARAMS_CH * mult for mult in VAE_PARAMS_CH_MULT]
#     down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
#     up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)
#
#     config = dict(
#         sample_size=VAE_PARAMS_RESOLUTION,
#         in_channels=VAE_PARAMS_IN_CH,
#         out_channels=VAE_PARAMS_OUT_CH,
#         down_block_types=tuple(down_block_types),
#         up_block_types=tuple(up_block_types),
#         block_out_channels=tuple(block_out_channels),
#         latent_channels=VAE_PARAMS_Z_CHANNELS,
#         layers_per_block=VAE_PARAMS_NUM_RES_BLOCKS,
#     )
#     return config
#
# # 初始化vae模型
# _, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path)
#
# vae_config = create_vae_diffusers_config()
# converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)
#
# vae = AutoencoderKL(**vae_config)
# info = vae.load_state_dict(converted_vae_checkpoint)
# print(info)
#
# # 读取一张图片并初始化
# import numpy as np
# from PIL import Image
#
# img_path = "../my_pic.jpg"
# img = Image.open(img_path)
# # img.show()
# img = np.array(img)
# img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
# # print(img.shape)
#
# latents = vae.encode(img).latent_dist.sample()
# # latents *= 0.1852
# print(latents)
#
# image = vae.decode(latents).sample
# image = (image / 2 + 0.5).clamp(0, 1)
# image = image.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
# image = (image * 255).round().astype("uint8")
# Image.fromarray(image).show()

import cv2

fcap = cv2.VideoCapture("../sports.mp4")

fps = fcap.get(cv2.CAP_PROP_FPS)
height = 256
width = 256
size = (height, width)

video_writer = cv2.VideoWriter("videoFrameTarget.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
success, frame_src = fcap.read()
while success:
    frame_target = frame_src[298:555, 111:368]
    video_writer.write(frame_target)

    success, frame_src = fcap.read()
fcap.release()




