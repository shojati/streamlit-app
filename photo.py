import os
import imageio
from tqdm import tqdm

def create_video(image_folder, video_name, fps=10):
    images = []
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        images.append(imageio.imread(image_path))

    output_video_path = os.path.join(image_folder, video_name)

    # Save video
    imageio.mimsave(output_video_path, images, fps=fps)

    print(f"Video created successfully: {output_video_path}")

# Example usage
image_folder_path = 'G:/Result3'
output_video_name = 'output_video3.mp4'
create_video(image_folder_path, output_video_name)
