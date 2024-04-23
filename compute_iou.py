import pandas as pd
from shapely.geometry import Polygon

# Lire le premier fichier CSV dans un DataFrame pandas
df1 = pd.read_csv("csvv.csv")

# Lire le deuxième fichier CSV dans un autre DataFrame pandas
df2 = pd.read_csv("csvv.csv")

# Créer des polygones pour le premier fichier
polygons1 = {}
current_id = None
current_points = []


def compute_iou(csv1, csv2):
    for idx, row in df1.iterrows():
        color = row['Color']
        if current_id is None:
            current_id = row['Color']
        if row['Color'] != current_id:
            if current_id not in polygons1:
                polygons1[current_id] = []
            if len(current_points) >= 3:
                polygons1[current_id].append(current_points)
            current_points = []
            current_id = row['Color']
        current_points.append((row['X'], row['Y']))

    if current_id not in polygons1:
        polygons1[current_id] = []
    if len(current_points) >= 3:
        polygons1[current_id].append(current_points)

    # Créer des polygones pour le deuxième fichier
    polygons2 = {}
    current_id = None
    current_points = []

    for idx, row in df2.iterrows():
        color = row['Color']
        if current_id is None:
            current_id = row['Color']
        if row['Color'] != current_id:
            if current_id not in polygons2:
                polygons2[current_id] = []
            if len(current_points) >= 3:
                polygons2[current_id].append(current_points)
            current_points = []
            current_id = row['Color']
        current_points.append((row['X'], row['Y']))

    if current_id not in polygons2:
        polygons2[current_id] = []
    if len(current_points) >= 3:
        polygons2[current_id].append(current_points)

    # Calculer l'intersection sur les polygones
    iou_results = {}
    for id1, points_list1 in polygons1.items():
        for id2, points_list2 in polygons2.items():
            # Comparer uniquement les polygones ayant le même ID
            if id1 == id2:
                for points1 in points_list1:
                    for points2 in points_list2:
                        if len(points1) >= 3 and len(points2) >= 3:
                            iou_results[(id1, id2)] = Polygon(points1).intersection(Polygon(points2)).area / Polygon(points1).union(Polygon(points2)).area

    # Afficher les résultats IoU
    for (id1, id2), iou in iou_results.items():
        print(f"IoU entre ID {id1} dans le premier fichier et ID {id2} dans le deuxième fichier: {iou}")


    # Calculer l'intersection sur les polygones et stocker le nombre de paires avec IoU supérieur à un seuil donné
    iou_threshold = 0.5
    total_pairs = 0
    successful_pairs = 0

    for id1, points_list1 in polygons1.items():
        for id2, points_list2 in polygons2.items():
            if id1 == id2:
                for points1 in points_list1:
                    for points2 in points_list2:
                        if len(points1) >= 3 and len(points2) >= 3:
                            total_pairs += 1
                            iou = Polygon(points1).intersection(Polygon(points2)).area / Polygon(points1).union(Polygon(points2)).area
                            if iou >= iou_threshold:
                                successful_pairs += 1

    # Calculer le pourcentage de réussite général de l'IoU
    success_rate = (successful_pairs / total_pairs) * 100 if total_pairs > 0 else 0

    print(f"Pourcentage de réussite général de l'IoU : {success_rate}%")