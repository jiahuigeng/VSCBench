import os
import pandas as pd


def check_image_centric():
    topics = ["discrimination", "drugs", "illegal_activity", "porngraphy", "religion", "violence"]

    df_input = pd.read_excel("vscbench_image_centric.xlsx")
    for idx, row in df_input.iterrows():
        category = row["Category"]
        img_cols = ["safe_img_0", "safe_img_1", "safe_img_2", "unsafe_img_0", "unsafe_img_1", "unsafe_img_2"]

        for item in img_cols:
            img_path = os.path.join("vscbench_image_centric_images", category, row[item])
            if not os.path.exists(img_path):
                print(f"{img_path} does not exist")




def check_text_centric():
    items = ["search_img", "typography_img", "concat_img", "figstep_img"]

    df_input = pd.read_excel("vscbench_text_centric.xlsx")
    for idx, row in df_input.iterrows():
        for item in items:
            sub_folder = item.split("_")[0]
            img_path = os.path.join("vscbench_text_centric_images", sub_folder, row[item])
            if not os.path.exists(img_path):
                print(f"{img_path} does not exist")



check_text_centric()
check_image_centric()



