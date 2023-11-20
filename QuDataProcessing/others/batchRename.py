import os


# Function to rename multiple files
def batchRename(filePath, oldName, newName):
    for count, filename in enumerate(os.listdir(filePath)):
        if oldName in filename:
            dst = filename.replace(oldName, newName)
            src = f"{filePath}/{filename}"
            dst = f"{filePath}/{dst}"

            os.rename(src, dst)


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    batchRename(r"L:\Data\Tree2.1\test\\", "1_", "10_")