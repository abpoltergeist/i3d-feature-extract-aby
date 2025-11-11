import os
import shutil

def organize_flow_images(root_folder):
    """
    Sorts flow images in 'videox_low_fps' folders into 'flow_x' and 'flow_y' subfolders.

    Args:
        root_folder (str): The path to the main folder containing all the
                           'videox_low_fps' directories.
    """
    print(f"Starting to process folders in: {root_folder}\n")

    # Check if the provided root folder actually exists
    if not os.path.isdir(root_folder):
        print(f"Error: The folder '{root_folder}' does not exist.")
        return

    try:
        # Loop through every item in the root folder
        for folder_name in os.listdir(root_folder):
            video_folder_path = os.path.join(root_folder, folder_name)

            # Check if it's a directory AND it matches the "videox_low_fps" pattern
            if (os.path.isdir(video_folder_path) and
                    folder_name.startswith('video') and
                    folder_name.endswith('_low_fps')):
                
                print(f"--- Processing folder: {folder_name} ---")

                # 1. Define the paths for the new 'flow_x' and 'flow_y' folders
                flow_x_path = os.path.join(video_folder_path, 'flow_x')
                flow_y_path = os.path.join(video_folder_path, 'flow_y')

                # 2. Create these new folders (if they don't already exist)
                os.makedirs(flow_x_path, exist_ok=True)
                os.makedirs(flow_y_path, exist_ok=True)

                file_count_x = 0
                file_count_y = 0

                # 3. Loop through all files inside the 'videox_low_fps' folder
                for filename in os.listdir(video_folder_path):
                    original_file_path = os.path.join(video_folder_path, filename)

                    # Make sure it's a file, not a directory
                    if os.path.isfile(original_file_path):
                        
                        # Use os.path.splitext to safely get the name before the extension
                        # This works for "flow_x_00001.jpg", "flow_y_00001.png", etc.
                        file_stem, file_ext = os.path.splitext(filename)

                        # 4. Check if the file name starts with 'flow_x'
                        if file_stem.startswith('flow_x'):
                            # This is an x-direction flow image
                            destination_path = os.path.join(flow_x_path, filename)
                            # Move the file
                            shutil.move(original_file_path, destination_path)
                            file_count_x += 1
                        
                        # 5. Check if the file name starts with 'flow_y'
                        elif file_stem.startswith('flow_y'):
                            # This is a y-direction flow image
                            destination_path = os.path.join(flow_y_path, filename)
                            # Move the file
                            shutil.move(original_file_path, destination_path)
                            file_count_y += 1

                print(f"Moved {file_count_x} files to 'flow_x'")
                print(f"Moved {file_count_y} files to 'flow_y'\n")

        print("--- All folders processed successfully! ---")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please check your folder permissions and paths.")

# This block runs the code when you execute the script directly
if __name__ == "__main__":
    # 1. Get the path from the user
    # .strip() removes any accidental leading/trailing whitespace
    path_input = input("Enter the path to the main folder containing your 'videox_low_fps' directories: ").strip()
    
    # 2. Run the organization function
    organize_flow_images(path_input)

