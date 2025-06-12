from azfuse import File
import os.path as op
import json

# recursively list all files under a folder
def get_all_files(file):
    entries = list(File.list(file))
    all_files = []
    for entry in entries:
        file_entries = list(File.list(entry))
        # print(entry, len(file_entries))
        if len(file_entries):
            # Recursively list files in the directory
            all_files += get_all_files(entry)
        else:
            # Add the file to the list
            all_files.append(entry)
    return all_files


def get_all_fileid_under_az_dir(folder, output_file=None, ext=".jpg"):
    files = get_all_files(folder)
    # print(str(files[1]).endswith(ext))
    # print(ext)
    print(len(files))
    all_file_ids = [f.replace(folder, "") for f in files if f.endswith(ext)]
    print(len(all_file_ids))
    if output_file:
        # split file into small chunks
        chunk_size = 5000
        all_file_ids_chunk = [all_file_ids[i:i + chunk_size] for i in range(0, len(all_file_ids), chunk_size)]
        for i, chunk in enumerate(all_file_ids_chunk):
            with File.open(output_file.replace(".json", f"_{i}.json"), "w") as f:
                json.dump(chunk, f)



if __name__ == "__main__":
    from fire import Fire
    Fire(get_all_fileid_under_az_dir)
