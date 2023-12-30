import os
import pdb


# Function to determine the accuracy category
def get_acc_category(acc):
    if acc == 1.0:
        return 'accuracy_1'
    elif 0.8 <= acc < 1.0:
        return 'accuracy_0.8-1'
    elif 0.5 <= acc < 0.8:
        return 'accuracy_0.5-0.8'
    elif 0.2 <= acc < 0.5:
        return 'accuracy_0.2-0.5'
    else:
        return 'accuracy_less_0.2'
    

import os

def create_main_page(acc_ranges, output_dir, experiment_name):
    html_content = f"""
    <html>
    <head>
        <title>{experiment_name} - Visualization of GT Images and Videos by Accuracy</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            a {{ color: #0275d8; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>{experiment_name}</h1>
        <h2>Select Accuracy Range for Visualization</h2>
    """

    for category in acc_ranges:
        html_content += f"<p><a href='{category}_page_1.html'>{category}</a></p>"

    html_content += """
    </body>
    </html>
    """

    with open(os.path.join(output_dir, 'index.html'), 'w') as file:
        file.write(html_content)


def create_accuracy_pages(videos_dir, images_dir, acc_ranges, output_dir, items_per_page=30):
    rel_videos_dir = os.path.relpath(videos_dir, output_dir)
    rel_images_dir = os.path.relpath(images_dir, output_dir)

    for category, files in acc_ranges.items():
        num_pages = len(files) // items_per_page + (1 if len(files) % items_per_page > 0 else 0)

        for page in range(num_pages):
            page_files = files[page * items_per_page:(page + 1) * items_per_page]
            page_number = page + 1
            html_content = f"""
            <html>
            <head>
                <title>Visualization - Accuracy Range {category} - Page {page_number}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    tr:hover {{ background-color: #f1f1f1; }}
                    img, video {{ max-width: 320px; height: auto; }}
                    .pagination a {{ padding: 8px 16px; margin: 0 4px; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>Visualization for Accuracy Range: {category} - Page {page_number}</h1>
                <table>
                    <tr><th>File Name</th><th>Ground Truth Image</th><th>Video</th><th>Accuracy</th></tr>
            """

            for base_name, video_file, acc in page_files:
                image_file = base_name + '.png'
                video_path = os.path.join(rel_videos_dir, video_file)
                image_path = os.path.join(rel_images_dir, image_file)

                if os.path.exists(os.path.join(images_dir, image_file)):
                    html_content += f"""
                        <tr>
                            <td>{base_name}</td>
                            <td><img src="{image_path}" /></td>
                            <td><video autoplay muted loop controls>
                                <source src="{video_path}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video></td>
                            <td>{acc}</td>
                        </tr>
                    """

            html_content += """
                </table>
                <div class='pagination'>
            """

            for p in range(1, num_pages + 1):
                html_content += f"<a href='{category}_page_{p}.html'>{p}</a> "

            html_content += """
                </div>
                <p><a href='index.html'>Back to Main Page</a></p>
            </body>
            </html>
            """

            page_filename = f'{category}_page_{page_number}.html'
            with open(os.path.join(output_dir, page_filename), 'w') as file:
                file.write(html_content)



def main():
    videos_dir = "./render_results/test_results"  # Directory containing videos
    images_dir = "./render_results/test_results"  # Directory containing ground truth images
    output_dir = "./render_results"  # Directory for output HTML files

    os.makedirs(output_dir, exist_ok=True)

    # Your existing code to categorize files into acc_ranges goes here
    acc_ranges = {
        'accuracy_1': [],
        'accuracy_0.8-1': [],
        'accuracy_0.5-0.8': [],
        'accuracy_0.2-0.5': [],
        'accuracy_less_0.2': []
    }


    # Sorting files into buckets
    for video_file in sorted(os.listdir(videos_dir)):
        if video_file.endswith('.mp4'):
            base_name, acc_str = os.path.splitext(video_file)[0].split('_')
            acc = float(acc_str)
            category = get_acc_category(acc)
            acc_ranges[category].append((base_name, video_file, acc))

    
    experiment_name = "standard training results"
    # Create main page and separate accuracy pages
    create_main_page(acc_ranges, output_dir, experiment_name)
    create_accuracy_pages(videos_dir, images_dir, acc_ranges, output_dir, 30)


if __name__ == "__main__":
    main()
