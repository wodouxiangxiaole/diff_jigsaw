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
    

def get_video_info(videos_dir, acc_category_function):
    video_info = {}
    for video_file in sorted(os.listdir(videos_dir)):
        if video_file.endswith('.mp4'):
            base_name, acc_str = os.path.splitext(video_file)[0].split('_')
            acc = float(acc_str)
            category = acc_category_function(acc)
            video_info[base_name] = (video_file, acc, category)
    return video_info

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
        html_content += f"<p><a href='html/{category}_page_1.html'>{category}</a></p>"

    html_content += """
    </body>
    </html>
    """

    with open(os.path.join(output_dir, 'index.html'), 'w') as file:
        file.write(html_content)



def create_accuracy_pages(videos_dir, images_dir, videos_dir_compare, acc_ranges, output_dir, items_per_page=30):
    linear_videos = get_video_info(videos_dir, get_acc_category)
    custom_videos = get_video_info(videos_dir_compare, get_acc_category)

    rel_images_dir = os.path.relpath(images_dir, output_dir)
    rel_videos_dir = os.path.relpath(videos_dir, output_dir)
    rel_videos_dir_compare = os.path.relpath(videos_dir_compare, output_dir)

    acc_ranges = {
        'accuracy_1': [],
        'accuracy_0.8-1': [],
        'accuracy_0.5-0.8': [],
        'accuracy_0.2-0.5': [],
        'accuracy_less_0.2': []
    }

    for base_name in linear_videos:
        if base_name in custom_videos:
            linear_video_file, linear_acc, _ = linear_videos[base_name]
            custom_video_file, custom_acc, custom_category = custom_videos[base_name]
            acc_ranges[custom_category].append((base_name, linear_video_file, linear_acc, custom_video_file, custom_acc))

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
                    <tr><th>File Name</th><th>Ground Truth Image</th><th>Linear Scheduler</th><th>Linear Scheduler Accuracy</th><th>Custom Scheduler</th><th>Custom Scheduler Accuracy</th></tr>
            """

            for item in page_files:
                base_name, linear_video_file, linear_acc, custom_video_file, custom_acc = item

                image_file = base_name + '.png'
                linear_video_path = os.path.join(rel_videos_dir, linear_video_file)
                custom_video_path = os.path.join(rel_videos_dir_compare, custom_video_file)
                image_path = os.path.join(rel_images_dir, image_file)

                if os.path.exists(os.path.join(images_dir, image_file)):
                    html_content += f"""
                        <tr>
                            <td>{base_name}</td>
                            <td><img src="{image_path}" /></td>
                            <td><video autoplay muted loop controls>
                                <source src="{linear_video_path}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video></td>
                            <td>{linear_acc}</td>
                            <td><video autoplay muted loop controls>
                                <source src="{custom_video_path}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video></td>
                            <td>{custom_acc}</td>
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
    videos_dir = "./render_results/linear_scheduler_results_bs50"  # Directory containing videos
    images_dir = "./render_results/linear_scheduler_results_bs50"  # Directory containing ground truth images
    videos_dir_compare = "./render_results/custom_scheduler_results_bs50" 
    output_dir = "./render_results"  # Directory for output HTML files
    html_dir = os.path.join(output_dir, "html")  # Subdirectory for HTML files

    os.makedirs(html_dir, exist_ok=True)
    
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

    
    experiment_name = "results"

    # Create main page and separate accuracy pages
    create_main_page(acc_ranges, output_dir, experiment_name)
    create_accuracy_pages(videos_dir, images_dir, videos_dir_compare, acc_ranges, html_dir, 30)

if __name__ == "__main__":
    main()
