import random


def read_data(max_data=None):
    with open("Selfie-dataset/new_selfie_dataset.txt") as file:
        text = file.read().split("\n")

    print(len(text), "datos encontrados.")
    # random.shuffle(text)

    if max_data:
        print("Cogiendo", max_data, "de ellos.")
        text = text[:max_data + 1]

    images_data = []
    for i, person in enumerate(text[:-1]):
        data = person.split(" ")
        info = {
            'file_name': data[0],
            'popularity_score': float(data[1]),
            'partial_faces': int(data[2]),
            'is_female': int(data[3]),
            'baby': int(data[4]),
            'child': int(data[5]),
            'teenager': int(data[6]),
            'youth': int(data[7]),
            'middle_age': int(data[8]),
            'senior': int(data[9]),
            'white': int(data[10]),
            'black': int(data[11]),
            'asian': int(data[12]),
            'oval_face': int(data[13]),
            'round_face': int(data[14]),
            'heart_face': int(data[15]),
            'smiling': int(data[16]),
            'mouth_open': int(data[17]),
            'frowning': int(data[18]),
            'wearing_glasses': int(data[19]),
            'wearing_sunglasses': int(data[20]),
            'wearing_lipstick': int(data[21]),
            'tongue_out': int(data[22]),
            'duck_face': int(data[23]),
            'black_hair': int(data[24]),
            'blond_hair': int(data[25]),
            'brown_hair': int(data[26]),
            'red_hair': int(data[27]),
            'curly_hair': int(data[28]),
            'straight_hair': int(data[29]),
            'braid_hair': int(data[30]),
            'showing_cellphone': int(data[31]),
            'using_earphone': int(data[32]),
            'using_mirror': int(data[33]),
            'braces': int(data[34]),
            'wearing_hat': int(data[35]),
            'harsh_lighting': int(data[36]),
            'dim_lighting': int(data[37])
        }
        images_data.append(info)

    return images_data
