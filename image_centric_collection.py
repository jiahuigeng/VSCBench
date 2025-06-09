import os
import pandas as pd
import argparse
import shutil
from utils_llm import get_commercial_model, prompt_commercial_model, get_open_model, get_open_model, prompt_open_model
from utils import init_df_cols, get_topic
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_row(idx, row, vscbench_img_path, model, args):
    # 定义每个线程处理的任务
    prompt = row["Prompt"]
    results = {}

    # if not pd.isna(row[f"safe_resp_{args.model_name}_0"]):
    #     return idx, results  # 如果已处理，跳过

    for j in range(3):
        # Safe image
        safe_img = row[f"safe_img_{j}"]
        safe_img_path = os.path.join(vscbench_img_path, safe_img)
        if pd.isna(row[f"safe_resp_{args.model_name}_{j}"]):
            safe_resp = prompt_commercial_model(model, args.model_name, prompt, safe_img_path)
            print(safe_resp)
            results[f"safe_resp_{args.model_name}_{j}"] = safe_resp
        else:
            results[f"safe_resp_{args.model_name}_{j}"] = row[f"safe_resp_{args.model_name}_{j}"]

        # Unsafe image
        unsafe_img = row[f"unsafe_img_{j}"]
        unsafe_img_path = os.path.join(vscbench_img_path, unsafe_img)

        if pd.isna(row[f"unsafe_resp_{args.model_name}_{j}"]):
            unsafe_resp = prompt_commercial_model(model, args.model_name, prompt, unsafe_img_path)
            print(unsafe_resp)
            results[f"unsafe_resp_{args.model_name}_{j}"] = unsafe_resp
        else:
            results[f"unsafe_resp_{args.model_name}_{j}"] = row[f"unsafe_resp_{args.model_name}_{j}"]

    return idx, results


def vscbench_collection(args):
    max_workers = 10
    file_input = args.file_eval
    topic = get_topic(file_input)
    vscbench_img_path = f"vscbench_image_centric_images/{topic}"

    save_xlsx = file_input.split(".")[0] + f"_{args.model_name}.xlsx"
    save_xlsx = os.path.join(f"results_{args.model_name}", save_xlsx)
    if not os.path.exists(save_xlsx):
        shutil.copyfile(file_input, save_xlsx)

    commercial_model = False
    if args.model_name in ["gpt4o", "claude", "gemini"]:
        commercial_model = True
        model = get_commercial_model(args.model_name)
        multi_thread = True
        if args.model_name == "claude":
            multi_thread = False

    else:
        model, tokenizer, processor = get_open_model(args.model_name)
        multi_thread = False

    df_input = pd.read_excel(save_xlsx)
    new_cols = []
    for j in range(3):
        new_cols.append(f"unsafe_resp_{args.model_name}_{j}")
        new_cols.append(f"safe_resp_{args.model_name}_{j}")
    df_input = init_df_cols(df_input, new_cols)

    ####
    # multi_thread = False
    if multi_thread:
        futures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, row in df_input.iterrows():
                print(idx)
                futures.append(executor.submit(process_row, idx, row, vscbench_img_path, model, args))

            for future in as_completed(futures):
                idx, results = future.result()
                for key, value in results.items():
                    df_input.at[idx, key] = value

        # Save the updated dataframe
        try:
            df_input.to_excel(save_xlsx, index=False)
        except:
            df_input.to_excel(save_xlsx.split(".")[0] + "_alt.xlsx", index=False)

    else:
        for idx, row in df_input.iterrows():
            # print(idx)
            prompt = row["Prompt"]
            # if not pd.isna(row[f"safe_resp_{args.model_name}_0"]):
            #     continue
            for j in range(3):
                safe_img = row[f"safe_img_{j}"]
                safe_img_path = os.path.join(vscbench_img_path, safe_img)
                if pd.isna(row[f"safe_resp_{args.model_name}_{j}"]):
                    if commercial_model:
                        safe_resp = prompt_commercial_model(model, args.model_name, prompt, safe_img_path)
                    else:
                        safe_resp = prompt_open_model(model, tokenizer, processor, args.model_name, prompt, safe_img_path)
                    print(safe_resp)
                    df_input.at[idx, f"safe_resp_{args.model_name}_{j}"] = safe_resp

                unsafe_img = row[f"unsafe_img_{j}"]
                unsafe_img_path = os.path.join(vscbench_img_path, unsafe_img)
                if pd.isna(row[f"unsafe_resp_{args.model_name}_{j}"]):
                    if commercial_model:
                        unsafe_resp = prompt_commercial_model(model, args.model_name, prompt, unsafe_img_path)
                    else:
                        unsafe_resp = prompt_open_model(model, tokenizer, processor, args.model_name, prompt,
                                                    unsafe_img_path)
                    print(unsafe_resp)
                    df_input.at[idx, f"unsafe_resp_{args.model_name}_{j}"] = unsafe_resp

            print(f"model_name: {args.model_name}; file_eval: {file_input}, index: {idx}")
            df_input.to_excel(save_xlsx, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_eval', type=str, default='vscbench_image_centric.xlsx')
    parser.add_argument('--model_name', type=str, default='gemini')

    args = parser.parse_args()

    vscbench_collection(args)
