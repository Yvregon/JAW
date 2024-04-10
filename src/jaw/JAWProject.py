import shutil
import os

"""
Script for JAW project fast generation.
"""

def snake_to_pascal_case(string: str):
    """Translate a snake_case string to a PascalCase string.
    :param string: String to translate.
    :type string: str.

    :returns: string to PascalCase format.
    """
    titled_string = string.title()
    space_joined_string = titled_string.replace('_', '')

    return space_joined_string


def generate_jaw_project(project_name: str, root_path: str):
    """
    Generate a new JAW project by copying base files from `__template__` folder then replace the placeholder names by the project name.
    :param project_name: The name of the project.
    :type project_name: str.
    :param root_path: The root directory of the project.
    :type root_path: str.

    :returns: None.


    """
    shutil.copytree(os.path.dirname(__file__) + "/__template__", root_path + "/", dirs_exist_ok=True)
    prefix: str = root_path + "/" + project_name

    os.rename(root_path + "/project_name", prefix)
    os.rename(prefix + "/project_name_trainer.py", prefix + "/" + project_name + "_trainer.py")

    search_text = "ProjectNameTrainer"
    replace_text = snake_to_pascal_case(project_name + "_trainer")
    
    with open(prefix + "/" + project_name + "_trainer.py", 'r') as file: 
        data = file.read() 
        data = data.replace(search_text, replace_text)

    with open(prefix + "/" + project_name + "_trainer.py", 'w') as file: 
        file.write(data) 


def main() -> None:
    """
    Handle the I/O process of the project generation.
    """
    current_folder = os.getcwd()
    
    project_name: str = input("Enter a project name : ")
    root_folder: str = input("Enter a root folde (default : " + current_folder + ") : ")
    root_folder = current_folder + "/" + root_folder

    generate_jaw_project(project_name, root_folder)

    print("New JAW project generated.")


if __name__ == "__main__":
    main()
