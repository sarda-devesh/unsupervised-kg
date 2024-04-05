import os

def replace():
    src_file = os.path.join("docker", "requirements.txt")
    dst_file = "requirements.txt"

    with open(src_file, 'r') as reader:
        lines = reader.readlines()
    
    with open(dst_file, 'w+') as writer:
        for line in lines:
            curr_line = line.strip()
            if "==" in curr_line:
                curr_line = curr_line[ : curr_line.index("==")]
            writer.write(curr_line + "\n")

if __name__ == "__main__":
    replace()