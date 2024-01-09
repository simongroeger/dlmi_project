
#
# Classificaion GUI
#
from classification_gui import *
def classification_gui():
    """Classification GUI"""

    path_1 = "Data/labelled_images/ampulla_of_vater/Ampulla_of_vater"
    path_2 = "Data/labelled_images/blood_-_fresh/Blood_fresh"
    data_path = "Data/labelled_images"
    metadata_path = "Data/metadata.csv"
    # dl_model = load_model()

    # image_data = load_data(path_1, path_2)
    root = create_window()
    create_caption(root)    
    image_data_field = create_image_data_field(root)
    neural_network_selected, baseline_selected = select_algorithm(root)
    prediction_field = create_prediction_field(root)
    # listbox = select_sample(root, image_data, metadata_path, image_data_field, prediction_field)
    sample_field = select_sample_from_file_explorer(root, data_path, metadata_path, image_data_field)
    classify_image(root, sample_field, neural_network_selected, baseline_selected, prediction_field)

    root.mainloop()

 
if __name__ == '__main__':
    classification_gui()