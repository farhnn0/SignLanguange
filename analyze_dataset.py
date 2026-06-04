import os

path = r"bisindo-kata-baru"
categories = ['5W+1H', 'Kata Ganti Orang', 'Kata Kerja', 'Kata Lainnya', 'Kata Sifat']
total_classes = 0
total_videos = 0
class_distribution = []

for cat in categories:
    cat_path = os.path.join(path, cat)
    if os.path.exists(cat_path):
        for label in os.listdir(cat_path):
            label_path = os.path.join(cat_path, label)
            if os.path.isdir(label_path):
                video_count = len([f for f in os.listdir(label_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
                print(f'{cat}/{label}: {video_count} videos')
                class_distribution.append(video_count)
                total_classes += 1
                total_videos += video_count

print(f'\n=== SUMMARY ===')
print(f'Total classes: {total_classes}')
print(f'Total videos: {total_videos}')
print(f'Average videos per class: {total_videos/total_classes:.1f}')
print(f'Min videos per class: {min(class_distribution)}')
print(f'Max videos per class: {max(class_distribution)}')
