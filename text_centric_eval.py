import os
from shutil import copyfile
import pandas as pd
import argparse
from utils import init_df_cols
from utils_eval import extract_gpt_label, gpt_eval
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def process_row(index, row, items, eval_resp_xlsx):
    print(index)
    query = row["prompt_origin"]

    for item in items:
        t_resp = row[f'resp_{item}']
        if pd.isna(t_resp):
            t_resp = 'Sorry, I cannot help with that request.'
        if not pd.isna(row[f'resp_{item}_gpt_label']) and not pd.isna(row[f'resp_{item}_gpt_eval']):
            continue

        eval_label, eval_context = gpt_eval(query, t_resp)
        print(eval_label, eval_context)

        row[f'resp_{item}_gpt_label'] = eval_label
        row[f'resp_{item}_gpt_eval'] = eval_context

    return index, row

def vsc_text_eval(args):

    max_workers = 10
    xlsx_resp_file = os.path.join(f"results_{args.model}", args.file_eval.split(".")[0] + f"_{args.model}.xlsx")
    if not xlsx_resp_file:
        print(f"{args.file_eval} not exists")

    eval_resp_xlsx = xlsx_resp_file.split(".")[0] + "_eval.xlsx"
    if not os.path.exists(eval_resp_xlsx):
        copyfile(xlsx_resp_file, eval_resp_xlsx)

    df_input = pd.read_excel(eval_resp_xlsx)

    items = ["txt", "search", "typography", "concat", 'figstep']
    resps = [f'resp_{item}' for item in items]
    new_cols = [f'resp_{item}_gpt_eval' for item in items] + [f'resp_{item}_gpt_label' for item in items]
    df_input = init_df_cols(df_input, new_cols)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda args: process_row(*args),
                                    [(index, row, items, eval_resp_xlsx)
                                     for index, row in df_input.iterrows()]))
    for index, row in results:
        df_input.iloc[index] = row

    df_input.to_excel(eval_resp_xlsx, index=False)

def vsc_text_stats(args):
    max_workers = 10
    xlsx_resp_file = os.path.join(f"results_{args.model}", args.file_eval.split(".")[0] + f"_{args.model}.xlsx")
    if not xlsx_resp_file:
        print(f"{args.file_eval} not exists")

    eval_resp_xlsx = xlsx_resp_file.split(".")[0] + "_eval.xlsx"
    if not os.path.exists(eval_resp_xlsx):
        copyfile(xlsx_resp_file, eval_resp_xlsx)

    df_input = pd.read_excel(eval_resp_xlsx)

    items = ["txt", "search", "typography", "concat", "figstep"]


    labels = []
    preds = {'txt': [], 'search': [], 'typography': [], 'concat': [], 'figstep': []}
    for index, row in df_input.iterrows():
        labels.append('contrast' in row['type'])
        for item in items:
            preds[item].append(row[f'resp_{item}_gpt_label'])

        if 'contrast' in row['type'] and not row[f'resp_txt_gpt_label']:
            print(index, row[f'resp_txt_gpt_label'])

    for key in preds:
        labels = np.array(labels)
        pred = np.array(preds[key])

        # 计算 True 部分的准确率
        true_indices = (labels == True)

        true_accuracy = np.mean(pred[true_indices] == labels[true_indices])

        # 计算 False 部分的准确率
        false_indices = (labels == False)

        false_accuracy = np.mean(pred[false_indices] == labels[false_indices])  #

        print(key)
        print(f"True 部分的准确率: {true_accuracy:.2f}")
        print(f"False 部分的准确率: {false_accuracy:.2f}")

    # for index, row in df_input.iterrows():
    #     query = row["prompt_origin"]
    #     for item in items:
    #         t_resp = row[f'resp_{item}']
    #         if not pd.isna(row[f'resp_{item}_gpt_label']) and not pd.isna(row[f'resp_{item}_gpt_eval']):
    #             continue
    #         # print(query, t_resp)
    #         eval_label, eval_context = gpt_eval(query, t_resp)
    #         print(eval_label, eval_context)
    #         df_input.at[index, f'resp_{item}_gpt_label'] = eval_label
    #         df_input.at[index, f'resp_{item}_gpt_eval'] = eval_context
    #
    #     df_input.to_excel(eval_resp_xlsx, index=False)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_eval', type=str, default='_centric.xlsx')
    parser.add_argument('--model', type=str, default='gemini')

    args = parser.parse_args()
    vsc_text_eval(args)
    vsc_text_stats(args)
