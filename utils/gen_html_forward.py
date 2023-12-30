import os

# Folder containing your MP4 files
folder_path = 'forward_translation/results'

# Start of the HTML file
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>My Videos</title>
    <style>
        table {
            width: 100%;
            table-layout: fixed;
        }
        video {
            width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <h1>My Video Collection</h1>
    <table>
"""

# Initialize a counter for videos per row
videos_in_row = 0

# Loop through each file in the folder
for file in os.listdir(folder_path):
    if file.endswith(".mp4"):
        # Start a new row if needed
        if videos_in_row == 0:
            html_content += "<tr>"

        video_path = os.path.join(folder_path, file)
        video_title = os.path.splitext(file)[0]

        # Add a video cell
        html_content += f"""
            <td>
                <h2>{video_title}</h2>
                <video autoplay muted loop controls>
                    <source src="{video_path}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </td>
        """

        videos_in_row += 1

        # Close the row if it has 4 videos
        if videos_in_row == 4:
            html_content += "</tr>\n"
            videos_in_row = 0

# Close the last row if it's not full
if videos_in_row > 0:
    html_content += "</tr>\n"

# End of the HTML file
html_content += """
    </table>
</body>
</html>
"""

# Save the HTML content to a file
with open('my_videos.html', 'w') as html_file:
    html_file.write(html_content)

print("HTML file generated successfully.")