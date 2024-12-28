def generate_output_path(input_path, applier):
    return input_path.replace("images", "outputs").replace(".", f"_{applier}.")