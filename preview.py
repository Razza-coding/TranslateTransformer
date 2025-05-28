def preview_raw_dataset(ds_raw, amount) -> None:
    for idx, item in enumerate(ds_raw):
        if idx + 1 == amount:
            break
        k = list(item.keys())
        if len(k) < 2:
            print(f"{'Unknown Item {idx}':<15} : {item}")
        else:
            lang_k = list(item[k[1]].keys())

            print(f"{k[0]:<15} : {item['id']}")
            print(f"{k[1]:<15} : ")
            print(f"\t{lang_k[0]:<10} : {item['translation'][lang_k[0]]}")
            print(f"\t{lang_k[1]:<10} : {item['translation'][lang_k[1]]}")