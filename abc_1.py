import os
import argparse
import cv2
import torch
import numpy as np
from gfpgan import GFPGANer
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer import CodeFormer

# Function for super-resolution using GFPGAN
def enhance_with_gfpgan(image):
    model = GFPGANer(model_path='path/to/GFPGAN.pth', upscale=2, arch='clean', channel_multiplier=2)
    _, _, restored_img = model.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    return restored_img

# Function for super-resolution using CodeFormer
def enhance_with_codeformer(image):
    model = CodeFormer()
    model.setup()
    restored_img = model.enhance(image, fidelity=0.7)  # 0.7 is a balanced fidelity
    return restored_img

# Function to enhance the lipsynced frame's subframe
def enhance_generated_part(input_frame, generated_part, superres_method):
    input_h, input_w = input_frame.shape[:2]
    gen_h, gen_w = generated_part.shape[:2]
    scale_ratio = max(input_h / gen_h, input_w / gen_w)

    if scale_ratio > 1:  # Only upscale if the generated part is of lower resolution
        if superres_method.lower() == "gfpgan":
            enhanced_part = enhance_with_gfpgan(generated_part)
        elif superres_method.lower() == "codeformer":
            enhanced_part = enhance_with_codeformer(generated_part)
        else:
            raise ValueError("Unsupported super-resolution method")
        return enhanced_part
    return generated_part  # No enhancement needed if resolutions match

# Function to process video frames and apply super-resolution
def process_video(input_video, input_audio, output_video, superres_method):
    video = cv2.VideoCapture(input_video)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {input_video}")
    for frame_idx in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        generated_part = frame[100:200, 100:200]  # Placeholder for generated part
        enhanced_part = enhance_generated_part(frame, generated_part, superres_method)
        frame[100:200, 100:200] = enhanced_part
        out_video.write(frame)
        print(f"Processed frame {frame_idx + 1}/{total_frames}")

    video.release()
    out_video.release()

    os.system(f"ffmpeg -i {output_video} -i {input_audio} -c:v copy -c:a aac -strict experimental {output_video}")
    print(f"Output video saved to: {output_video}")

# Main function for CLI
def main():
    parser = argparse.ArgumentParser(description="MuseTalk video enhancement with super-resolution")
    parser.add_argument('--superres', type=str, required=True, choices=['GFPGAN', 'CodeFormer'],
                        help="Choose the super-resolution method (GFPGAN/CodeFormer)")
    parser.add_argument('-iv', '--input_video', type=str, required=True, help="Path to input video")
    parser.add_argument('-ia', '--input_audio', type=str, required=True, help="Path to input audio")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to save output video")

    args = parser.parse_args()
    process_video(args.input_video, args.input_audio, args.output, args.superres)

if __name__ == "__main__":
    main()
