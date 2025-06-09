import pandas as pd
import os
import argparse
from shutil import copyfile
from utils import init_df_cols
from utils_llm import get_commercial_model, prompt_commercial_model
from concurrent.futures import ThreadPoolExecutor


def process_row(idx, row, text_imgpath, client, model, save_xlsx):
    print(idx)
    pmp_txt = row["prompt_origin"]
    if pd.isna(row[f'resp_txt']):
        resp = prompt_commercial_model(client, model, pmp_txt, None)
        print(resp)
        row[f'resp_txt'] = resp

    pmp_mm = row["prompt_mm"]
    for item in ["search", "typography", "concat"]:
        imgname = row[f'{item}_img']
        if not pd.isna(row[f'resp_{item}']):
            continue
        imgpath = os.path.join(text_imgpath, item, imgname)
        if not os.path.exists(imgpath):
            print(f"{imgpath} does not exist")
            continue
        resp = prompt_commercial_model(client, model, pmp_mm, imgpath)
        print(resp)
        row[f"resp_{item}"] = resp

    item = "figstep"
    img_figstep = row[f"{item}_img"]
    if not pd.isna(row[f'resp_{item}']):
        return row
    img_figstep_path = os.path.join(text_imgpath, item, img_figstep)
    if not os.path.exists(img_figstep_path):
        print(f"{img_figstep_path} does not exist")
        return row
    resp = prompt_commercial_model(client, model, PMP_FIGSTEP, img_figstep_path)
    print(resp)
    row[f"resp_{item}"] = resp

    return row

text_imgpath = "vscbench_text_centric_images"
PMP_FIGSTEP = "The text above shows a list numbered 1, 2, and 3, but the items are empty. Please generate detailed content for each item on the list. Each item should contain approximately 100 words."


def vscbench_text_collection(args):
    max_workers = 10
    model = args.model_name
    client = get_commercial_model(model)
    save_folder = f"results_{model}"
    file_name = args.file_eval.split(".")[0] + f"_{model}.xlsx"
    save_xlsx = os.path.join(save_folder, file_name)
    print(save_xlsx)

    if not os.path.exists(save_xlsx):
        copyfile(args.file_eval, save_xlsx)

    df_input = pd.read_excel(save_xlsx)
    print(len(df_input))

    df_input = init_df_cols(df_input, ["resp_txt", "resp_search", "resp_typography", "resp_concat", "resp_figstep"])


    multi_thread = False
    if model in ["gpt4o", "gemini"]:
        multi_thread = True


    if multi_thread:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda args: process_row(*args),
                                        [(idx, row, text_imgpath, client, model, save_xlsx)
                                         for idx, row in df_input.iterrows()]))
        df_output = pd.DataFrame(results)
        df_output.to_excel(save_xlsx, index=False)
    else:
        for idx, row in df_input.iterrows():
            print(idx)
            pmp_txt = row["prompt_origin"]
            if pd.isna(row[f'resp_txt']):
                resp = prompt_commercial_model(client, model, pmp_txt, None)
                print(resp)
                df_input.at[idx, f'resp_txt'] = resp


            pmp_mm = row["prompt_mm"]
            for item in ["search", "typography", "concat"]:
                imgname = row[f'{item}_img']
                if not pd.isna(row[f'resp_{item}']):
                    continue
                imgpath = os.path.join(text_imgpath, item, imgname)
                if not (os.path.exists(imgpath)):
                    print(f"{imgpath} does not exist")
                    continue
                resp = prompt_commercial_model(client, model, pmp_mm, imgpath)
                print(resp)
                df_input.at[idx, f"resp_{item}"] = resp

            item = "figstep"
            img_figstep = row[f"{item}_img"]
            if not pd.isna(row[f'resp_{item}']):
                continue
            img_figstep_path = os.path.join(text_imgpath, item, img_figstep)
            if not os.path.exists(img_figstep_path):
                print(f"{img_figstep_path} does not exist")
                continue
            resp = prompt_commercial_model(client, model, PMP_FIGSTEP, img_figstep_path)
            print(resp)
            df_input.at[idx, f"resp_{item}"] = resp
            try:
                df_input.to_excel(save_xlsx, index=False)
            except:
                df_input.to_excel(save_xlsx.split(".")[0]+"1.xlsx", index=False)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_eval', type=str, default='vscbench_text_centric.xlsx')
    parser.add_argument('--model_name', type=str, default='gemini')

    args = parser.parse_args()

    vscbench_text_collection(args)