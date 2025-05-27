def preview_raw_dataset(ds_raw, amount) -> None:
    for idx, item in enumerate(ds_raw):
        if idx + 1 == amount:
            break
        print(f"[ Item {idx + 1} ]")
        for k in item:
            print(f"\t{k:<10} : {item[k]}")


#preview_raw_dataset("LLaMAX/BenchMAX_General_Translation", "ted_en")
#preview_raw_dataset("LLaMAX/BenchMAX_General_Translation", "ted_zh")